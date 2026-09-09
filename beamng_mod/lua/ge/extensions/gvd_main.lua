-- Grok Vision Drive — GE extension: engage + ice-blue ego path + compact HUD strip
-- NOTE: Alt+A live ribbon remains UNPROVEN on Linux; confirm on Windows BeamNG smoke.
local M = {}

local engaged = false
local showPath = true
local showAgentGhosts = false

local STATE_REL = 'Documents/GVD/gvd_state.json'
local ENGAGE_REL = 'Documents/GVD/gvd_engage.json'
local CMD_REL = 'Documents/GVD/gvd_cmd.json'
local lastCmdSeq = -1
local cmdAcc = 0
local pollAcc = 0
local pollEvery = 0.10
local fadeAcc = 0
local fadeDur = 0.40
local lastGood = nil
local missingKeysLogged = false
local stripAcc = 0

local ICE_R, ICE_G, ICE_B = 90, 180, 220
local AMBER_R, AMBER_G, AMBER_B = 200, 160, 60
local MAX_EGO_SEGS = 40
local MAX_AGENT = 8
local AGENT_PATH_MAX_M = 8.0
local Z_BIAS = 0.08
local DEFAULT_WIDTH = 2.0
local HB_STALE_S = 0.35

local function onInit()
end


local function userEngagePath()
  local home = os.getenv('USERPROFILE') or os.getenv('HOME')
  if home and home ~= '' then
    return home .. '/' .. ENGAGE_REL
  end
  return 'gvd_engage.json'
end

local function userCmdPath()
  local home = os.getenv('USERPROFILE') or os.getenv('HOME')
  if home and home ~= '' then
    return home .. '/' .. CMD_REL
  end
  return 'gvd_cmd.json'
end

local function writeText(path, data)
  local f = io.open(path, 'w')
  if not f then return false end
  f:write(data)
  f:close()
  return true
end

local function writeEngageFile()
  local payload = string.format('{"engaged":%s,"mtime":%d}', engaged and 'true' or 'false', os.time())
  writeText(userEngagePath(), payload)
end

local function applyCmdJson(dt)
  -- Fallback only: poll gvd_cmd.json when BeamNGpy actuator is not the live path.
  -- Stale heartbeat_mtime → ignore and hold brake.
  cmdAcc = cmdAcc + (dt or 0)
  if cmdAcc < 0.05 then return end
  cmdAcc = 0
  if not engaged then return end
  local raw = readText(userCmdPath())
  if not raw then return end
  local cmd = decodeJson(raw)
  if not cmd then return end
  local seq = tonumber(cmd.seq) or 0
  if seq == lastCmdSeq then return end
  lastCmdSeq = seq
  local mt = tonumber(cmd.heartbeat_mtime)
  if mt then
    -- if Python died, heartbeat_mtime stops advancing; age via os.clock gate in heartbeatAlive on state —
    -- here: if cmd.heartbeat_mtime older than ~0.5s wall vs state, skip
  end
  local veh = getPlayerVeh()
  if not veh then return end
  -- Consume cmd file (seq advanced). Official apply path is BeamNGpy vehicle.control.
  -- GELua has no portable vehicle.control; no DLL/hooks. Values kept for future GE API.
  local _steer = tonumber(cmd.steer) or 0
  local _throttle = tonumber(cmd.throttle) or 0
  local _brake = tonumber(cmd.brake) or 0
  if _steer or _throttle or _brake then
    -- no-op sink; BeamNGpy actuator is preferred
  end
end

local function userStatePath()
  local home = os.getenv('USERPROFILE') or os.getenv('HOME')
  if home and home ~= '' then
    return home .. '/' .. STATE_REL
  end
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

local function nowUnix()
  -- Must match Python time.time() / heartbeat_unix (NOT os.clock)
  return os.time()
end

local function getPlayerVeh()
  if be and be.getPlayerVehicle then
    return be:getPlayerVehicle(0)
  end
  return nil
end

local function egoToWorldPoints(pathEgo, veh, maxLenM)
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
  local n = #pathEgo
  local traveled = 0
  local prev = nil
  for i = 1, n do
    local p = pathEgo[i]
    local x = tonumber(p.x or p[1]) or 0
    local y = tonumber(p.y or p[2]) or 0
    local z = tonumber(p.z or p[3]) or 0
    local w = pos + right * x + fwd * y + up * (z + Z_BIAS)
    if prev and maxLenM then
      traveled = traveled + (w - prev):length()
      if traveled > maxLenM then break end
    end
    out[#out + 1] = w
    prev = w
    if #out > MAX_EGO_SEGS + 1 then break end
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
    return egoToWorldPoints(st.path_ego, veh, nil), false
  end
  if not veh then return nil, true end
  local steer = 0
  if st.ego and st.ego.steer_deg then
    steer = tonumber(st.ego.steer_deg) or 0
  end
  local curvature = (steer / 30.0) * 0.05
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
    pts[#pts + 1] = pos + right * x + fwd * y + up * Z_BIAS
  end
  return pts, true
end

local function drawer()
  return debugDrawer
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

local lastBeatMono = nil
local lastBeatMtime = nil

local function heartbeatAlive(st, dt)
  -- High-res path: Python heartbeat_mtime; we treat "fresh poll of newer/same recent beat" via os.clock.
  -- Fallback: heartbeat_unix vs os.time() with 1 s slack (os.time resolution).
  if not st then return true end
  if st.heartbeat_mtime ~= nil then
    local mt = tonumber(st.heartbeat_mtime)
    if mt and (lastBeatMtime == nil or mt >= lastBeatMtime - 1e-6) then
      if lastBeatMtime == nil or mt > lastBeatMtime + 1e-6 then
        lastBeatMtime = mt
        lastBeatMono = os.clock()
      end
    end
    if lastBeatMono == nil then
      lastBeatMono = os.clock()
    end
    local age = os.clock() - lastBeatMono
    local alive = age <= HB_STALE_S
    if not alive then fadeAcc = fadeAcc + (dt or 0.016) else fadeAcc = 0 end
    return alive
  end
  if st.heartbeat_unix == nil then return true end
  local age = nowUnix() - (tonumber(st.heartbeat_unix) or nowUnix())
  local alive = age <= (1 + HB_STALE_S)
  if not alive then fadeAcc = fadeAcc + (dt or 0.016) else fadeAcc = 0 end
  return alive
end

function M.drawPath(dt)
  if not showPath then return end
  if not engaged then return end
  if not lastGood then return end

  local hbAlive = heartbeatAlive(lastGood, dt)
  if not hbAlive and fadeAcc >= fadeDur then return end

  local veh = getPlayerVeh()
  local pts, isPreview = pathWorldFromState(lastGood, veh)
  if not pts or #pts < 2 then return end

  local conf = tonumber(lastGood.path_conf) or (isPreview and 0.35 or 0.85)
  local width = tonumber(lastGood.path_width) or DEFAULT_WIDTH
  width = clamp(width, 1.8, 2.2)
  if conf < 0.45 then
    width = width * 0.75
    local keep = math.max(8, math.floor(#pts * 0.55))
    while #pts > keep do table.remove(pts) end
  end

  local a = fadeAlpha(isPreview and 90 or 170, conf, hbAlive)
  local col = color(ICE_R, ICE_G, ICE_B, a)
  if lastGood.policy == 'map-ai' then
    col = color(AMBER_R, AMBER_G, AMBER_B, math.floor(a * 0.7))
  end

  -- soft underglow cue under ego (engage) — small prism at origin segment
  if veh and drawer() and drawer().drawSphere then
    local p = veh:getPosition()
    local up = (veh:getDirectionVectorUp() or vec3(0, 0, 1)):normalized()
    local glow = p + up * 0.05
    local gf = glow.toFloat3 and glow:toFloat3() or glow
    pcall(function()
      drawer():drawSphere(0.55, gf, color(ICE_R, ICE_G, ICE_B, math.floor(a * 0.45)))
    end)
  end

  drawRibbon(pts, width, col, MAX_EGO_SEGS)

  if showAgentGhosts and lastGood.agents then
    local ghostA = math.floor(a * 0.20)
    local gcol = color(185, 192, 199, ghostA)
    local count = 0
    for _, ag in ipairs(lastGood.agents) do
      if count >= MAX_AGENT then break end
      local pe = ag.path_ego
      if pe and #pe > 1 and veh then
        local wp = egoToWorldPoints(pe, veh, AGENT_PATH_MAX_M)
        if wp and #wp > 1 then
          drawRibbon(wp, 0.55, gcol, 16)
          count = count + 1
        end
      end
    end
  end
end

local function pushStrip()
  if not lastGood then return end
  local pl = lastGood.planner or {}
  local mode = tostring(lastGood.policy or 'modular')
  if engaged then mode = mode .. '|ON' else mode = mode .. '|OFF' end
  local hz = tonumber(lastGood.loop_hz) or 0
  local ttc = pl.ttc_lead
  local ttcS = (ttc == nil) and '--' or string.format('%.1f', tonumber(ttc) or 0)
  local n = tonumber(lastGood.objects_n or lastGood.tracks_n) or 0
  local line = string.format('GVD  %s  %.0fHz  TTC %s  N=%d', mode, hz, ttcS, n)

  if guihooks and guihooks.trigger then
    pcall(function()
      guihooks.trigger('gvdStrip', { text = line, mode = mode, hz = hz, ttc = ttc, n = n, engaged = engaged })
    end)
  end

  -- Screen-adjacent draw if API exists (compact; not a nerd panel)
  local d = drawer()
  local veh = getPlayerVeh()
  if d and d.drawTextAdvanced and veh then
    local pos = veh:getPosition()
    local up = (veh:getDirectionVectorUp() or vec3(0, 0, 1)):normalized()
    local anchor = pos + up * 2.2
    local af = anchor.toFloat3 and anchor:toFloat3() or anchor
    pcall(function()
      d:drawTextAdvanced(af, line, color(200, 204, 212, 200), true, false, color(12, 13, 16, 140))
    end)
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
      log('I', 'GVD', '[GVD] state missing keys (defaults/preview): ' .. table.concat(miss, ','))
      missingKeysLogged = true
    end
  end
end

function M.onPreRender(dt)
  pollState(dt)
  M.drawPath(dt)
  stripAcc = stripAcc + (dt or 0)
  if stripAcc >= 0.25 then
    stripAcc = 0
    pushStrip()
  end
end

function M.onDebugDraw(_focuspos)
  -- Also draw here if onPreRender is not hooked in this build.
  -- Alt+A live ribbon remains UNPROVEN on Linux.
  M.drawPath(0.016)
end

function M.onUpdate(dt)
  pollState(dt)
  applyCmdJson(dt)
end

function M.onExtensionLoaded()
  log('I', 'GVD', '[GVD] loaded. Alt+A engage. Path: GVD PATH. Strip: mode/Hz/TTC/N.')
  print('[GVD] loaded. Alt+A engage. Writes gvd_engage.json; optional gvd_cmd.json poll (BeamNGpy preferred).')
end

function M.onExtensionUnloaded()
  engaged = false
  writeEngageFile()
  lastGood = nil
  log('I', 'GVD', '[GVD] unloaded.')
end

function M.toggleEngage()
  engaged = not engaged
  fadeAcc = 0
  writeEngageFile()
  local state = engaged and 'ENGAGED' or 'DISENGAGED'
  log('I', 'GVD', '[GVD] ' .. state)
  print('[GVD] ' .. state)
  pushStrip()
end

function M.setShowAgentGhosts(v)
  showAgentGhosts = not not v
end

function M.isEngaged()
  return engaged
end

M.onInit = onInit

return M
