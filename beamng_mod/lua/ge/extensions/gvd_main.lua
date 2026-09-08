-- Grok Vision Drive — GE extension: engage + ice-blue ego path ribbon (no DLL inject)
local M = {}

local engaged = false
local showPath = true
local showAgentGhosts = false -- off by default; GVD app toggle later

local STATE_REL = 'Documents/GVD/gvd_state.json'
local pollAcc = 0
local pollEvery = 0.10
local fadeAcc = 0
local fadeDur = 0.40
local lastHb = 0
local lastGood = nil -- {path_world=..., path_width=..., path_conf=..., agents=..., debug_preview=bool}
local missingKeysLogged = false

local ICE_R, ICE_G, ICE_B = 90, 180, 220
local AMBER_R, AMBER_G, AMBER_B = 200, 160, 60
local MAX_EGO_SEGS = 40
local MAX_AGENT = 8
local MAX_AGENT_SEGS = 10
local Z_BIAS = 0.08
local DEFAULT_WIDTH = 2.0

local function onInit()
end

local function userStatePath()
  local home = os.getenv('USERPROFILE') or os.getenv('HOME')
  if home and home ~= '' then
    return home .. '/' .. STATE_REL
  end
  -- BeamNG user folder fallback (may sit next to mods)
  if FS and FS.getUserPath then
    return FS:getUserPath() .. 'settings/gvd_state.json'
  end
  return 'gvd_state.json'
end

local function readText(path)
  if readFile then
    local ok, data = pcall(readFile, path)
    if ok and data and data ~= '' then return data end
  end
  if FS and FS.readFile then
    local ok, data = pcall(function() return FS:readFile(path) end)
    if ok and data and data ~= '' then return data end
  end
  local f = io.open(path, 'r')
  if not f then return nil end
  local data = f:read('*a')
  f:close()
  return data
end

local function decodeJson(s)
  if not s then return nil end
  if jsonDecode then
    local ok, t = pcall(jsonDecode, s)
    if ok then return t end
  end
  if util_jsonDecode then
    local ok, t = pcall(util_jsonDecode, s)
    if ok then return t end
  end
  return nil
end

local function clamp(x, a, b)
  if x < a then return a end
  if x > b then return b end
  return x
end

local function getPlayerVeh()
  if be and be.getPlayerVehicle then
    return be:getPlayerVehicle(0)
  end
  return nil
end

-- BeamNG vehicle frame: +X right, +Y forward, +Z up → world via pos + dir/right
local function egoToWorldPoints(pathEgo, veh)
  if not pathEgo or not veh then return nil end
  local pos = veh:getPosition()
  local fwd = veh:getDirectionVector()
  local up = veh:getDirectionVectorUp() or vec3(0, 0, 1)
  local right = fwd:cross(up)
  if right:length() < 1e-6 then
    right = vec3(1, 0, 0)
  else
    right = right:normalized()
  end
  fwd = fwd:normalized()
  up = up:normalized()

  local out = {}
  local n = math.min(#pathEgo, MAX_EGO_SEGS + 1)
  for i = 1, n do
    local p = pathEgo[i]
    local x = tonumber(p.x or p[1]) or 0
    local y = tonumber(p.y or p[2]) or 0
    local z = tonumber(p.z or p[3]) or 0
    local w = pos + right * x + fwd * y + up * (z + Z_BIAS)
    out[#out + 1] = w
  end
  return out
end

local function pathWorldFromState(st, veh)
  if st.path_world and #st.path_world > 1 then
    local out = {}
    local n = math.min(#st.path_world, MAX_EGO_SEGS + 1)
    for i = 1, n do
      local p = st.path_world[i]
      out[#out + 1] = vec3(tonumber(p.x or p[1]) or 0, tonumber(p.y or p[2]) or 0, (tonumber(p.z or p[3]) or 0) + Z_BIAS)
    end
    return out, false
  end
  if st.path_ego and #st.path_ego > 1 and veh then
    return egoToWorldPoints(st.path_ego, veh), false
  end
  -- debug preview: short straight from steer (or ahead)
  if not veh then return nil, true end
  local steer = 0
  if st.ego and st.ego.steer_deg then
    steer = tonumber(st.ego.steer_deg) or 0
  end
  local curvature = (steer / 30.0) * 0.05 -- mild
  local pts = {}
  local pos = veh:getPosition()
  local fwd = veh:getDirectionVector():normalized()
  local up = (veh:getDirectionVectorUp() or vec3(0, 0, 1)):normalized()
  local right = fwd:cross(up):normalized()
  local heading = 0
  local x, y = 0.0, 0.0
  for i = 0, 15 do
    y = y + 1.0
    heading = heading + curvature
    x = x + math.sin(heading)
    local w = pos + right * x + fwd * y + up * Z_BIAS
    pts[#pts + 1] = w
  end
  return pts, true
end

local function drawer()
  if debugDrawer then return debugDrawer end
  return nil
end

local function drawRibbon(points, width, col, segsCap)
  local d = drawer()
  if not d or not points or #points < 2 then return end
  local n = math.min(#points - 1, segsCap or MAX_EGO_SEGS)
  local half = float3 and float3(0.05, width, 0.05) or nil

  for i = 1, n do
    local a = points[i]
    local b = points[i + 1]
    local af = a.toFloat3 and a:toFloat3() or a
    local bf = b.toFloat3 and b:toFloat3() or b
    local drew = false
    if half and d.drawSquarePrism then
      local ok = pcall(function()
        d:drawSquarePrism(af, bf, half, half, col)
      end)
      drew = ok
    end
    if not drew and d.drawLine then
      pcall(function() d:drawLine(af, bf, col) end)
    end
  end
end

local function fadeAlpha(baseA, conf, hbAlive)
  local a = baseA * clamp(conf or 1, 0.15, 1.0)
  if not hbAlive then
    local t = clamp(1.0 - (fadeAcc / fadeDur), 0, 1)
    a = a * t
  end
  return math.floor(clamp(a, 0, 255))
end

function M.drawPath(_dt)
  if not showPath then return end
  if not engaged then return end
  if not lastGood then return end

  local now = os.clock()
  local hb = tonumber(lastGood.heartbeat_ms) or tonumber(lastGood.hb) or 0
  -- treat missing heartbeat as alive for stub; if field present and stale >300ms, fade
  local hbAlive = true
  if lastGood.heartbeat_unix then
    local age = now - (tonumber(lastGood.heartbeat_unix) or now)
    hbAlive = age < 0.35
    if not hbAlive then
      fadeAcc = fadeAcc + (_dt or 0.016)
      if fadeAcc >= fadeDur then return end
    else
      fadeAcc = 0
    end
  end

  local veh = getPlayerVeh()
  local pts, isPreview = pathWorldFromState(lastGood, veh)
  if not pts or #pts < 2 then return end

  local conf = tonumber(lastGood.path_conf) or (isPreview and 0.35 or 0.85)
  local width = tonumber(lastGood.path_width) or DEFAULT_WIDTH
  width = clamp(width, 0.8, 2.4)
  if conf < 0.45 then
    width = width * 0.7
    -- shorter ribbon when low conf
    local keep = math.max(8, math.floor(#pts * 0.55))
    while #pts > keep do table.remove(pts) end
  end

  local a = fadeAlpha(isPreview and 90 or 160, conf, hbAlive)
  local col = color(ICE_R, ICE_G, ICE_B, a)
  if lastGood.policy == 'map-ai' then
    col = color(AMBER_R, AMBER_G, AMBER_B, math.floor(a * 0.7))
  end

  drawRibbon(pts, width, col, MAX_EGO_SEGS)

  -- optional agent mode-0 ghosts (dim)
  if showAgentGhosts and lastGood.agents then
    local ghostA = math.floor(a * 0.20)
    local gcol = color(185, 192, 199, ghostA)
    local count = 0
    for _, ag in ipairs(lastGood.agents) do
      if count >= MAX_AGENT then break end
      local pe = ag.path_ego
      if pe and #pe > 1 and veh then
        local wp = egoToWorldPoints(pe, veh)
        if wp then
          while #wp > MAX_AGENT_SEGS + 1 do table.remove(wp) end
          drawRibbon(wp, 0.6, gcol, MAX_AGENT_SEGS)
          count = count + 1
        end
      end
    end
  end
end

local function pollState(dt)
  pollAcc = pollAcc + (dt or 0)
  if pollAcc < pollEvery then return end
  pollAcc = 0
  local raw = readText(userStatePath())
  if not raw then return end
  local st = decodeJson(raw)
  if not st then return end
  lastGood = st
  if st.gvd_show_path ~= nil then showPath = not not st.gvd_show_path end
  if st.show_agent_ghosts ~= nil then showAgentGhosts = not not st.show_agent_ghosts end
  if not missingKeysLogged then
    local miss = {}
    if not st.path_ego and not st.path_world then miss[#miss + 1] = 'path_ego|path_world' end
    if not st.path_width then miss[#miss + 1] = 'path_width' end
    if not st.path_conf then miss[#miss + 1] = 'path_conf' end
    if #miss > 0 then
      log('I', 'GVD', '[GVD] state missing keys (using defaults/preview): ' .. table.concat(miss, ','))
      missingKeysLogged = true
    end
  end
end

function M.onPreRender(dt)
  pollState(dt)
  M.drawPath(dt)
end

function M.onDebugDraw(_focuspos)
  -- also draw here if onPreRender is not hooked in this build
  M.drawPath(0.016)
end

function M.onUpdate(dt)
  pollState(dt)
end

function M.onExtensionLoaded()
  log('I', 'GVD', '[GVD] loaded. Alt+A engage. Path ribbon: GVD PATH (ice-blue).')
  print('[GVD] loaded. Alt+A engage. Path ribbon reads Documents/GVD/gvd_state.json')
end

function M.onExtensionUnloaded()
  engaged = false
  lastGood = nil
  log('I', 'GVD', '[GVD] unloaded.')
end

function M.toggleEngage()
  engaged = not engaged
  fadeAcc = 0
  local state = engaged and 'ENGAGED' or 'DISENGAGED'
  log('I', 'GVD', '[GVD] ' .. state)
  print('[GVD] ' .. state)
end

function M.setShowAgentGhosts(v)
  showAgentGhosts = not not v
end

function M.isEngaged()
  return engaged
end

M.onInit = onInit

return M
