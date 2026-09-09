-- Grok Vision Drive — GE extension: engage + ice-blue ego path + compact HUD strip
-- NOTE: Alt+A live ribbon remains UNPROVEN on Linux; confirm on Windows BeamNG smoke.
local M = {}

local engaged = false
local showPath = true
local showAgentGhosts = false
local showScene = true       -- in-app VISION canvas (player can switch it off; shares the GPU with BeamNG)
local policyReq = nil        -- session-scoped policy request for the Python supervisor
local vizScreenReq = nil     -- session-scoped GVD VISION monitor request

local STATE_REL = 'Documents/GVD/gvd_state.json'
local ENGAGE_REL = 'Documents/GVD/gvd_engage.json'
local CMD_REL = 'Documents/GVD/gvd_cmd.json'
local UI_PREFS_REL = 'Documents/GVD/gvd_ui_prefs.json'
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
local MAX_AGENT = 8  -- forecast ribbons; track hulls use MAX_TRACK_GHOSTS
local AGENT_PATH_MAX_M = 8.0
local Z_BIAS = 0.10  -- dual-viz: reduce z-fight on pavement
local DEFAULT_WIDTH = 2.0
local MAX_TRACK_GHOSTS = 16
local HB_STALE_S = 0.35

-- Caps for the geometry we hand to the in-game app (keep the guihooks payload small)
local UI_PATH_PTS = 28
local UI_TRACKS = 12
local UI_LANES = 3
local UI_LANE_PTS = 12
local CAM_IDS = { 'narrow', 'main', 'wide', 'pillarL', 'pillarR', 'repeatL', 'repeatR', 'rear' }

local function onInit()
end



-- Resolve Documents/GVD without relying on USERPROFILE (empty in GELua on some Windows builds).
-- Live bug: bare gvd_engage.json landed under BeamNG userfolder current\, Python read Documents\GVD\.
local gvdDocsResolved = nil
local gvdDocsLogged = false

local function _stripToUserHome(p)
  if not p or p == '' then return nil end
  p = tostring(p):gsub('\\', '/')
  -- C:/Users/Name/AppData/Local/... → C:/Users/Name
  local home = p:match('^(.+)/AppData/Local') or p:match('^(.+)/AppData/Roaming') or p:match('^(.+)/AppData')
  if home and home ~= '' then return home end
  return nil
end

local function _tryHomeEnv()
  local home = os.getenv('USERPROFILE') or os.getenv('HOME')
  if home and home ~= '' then return home end
  local localApp = os.getenv('LOCALAPPDATA')
  if localApp and localApp ~= '' then
    local h = _stripToUserHome(localApp)
    if h then return h end
  end
  return nil
end

local function _tryFsHome()
  if FS then
    if FS.getUserPath then
      local ok, up = pcall(function() return FS:getUserPath() end)
      if ok and up and up ~= '' then
        local h = _stripToUserHome(up)
        if h then return h end
      end
    end
    -- virtual2Native / getFileRealPath on a known user-folder VFS path
    for _, vp in ipairs({'settings', '/settings', 'settings/'}) do
      if FS.virtual2Native then
        local ok, native = pcall(function() return FS:virtual2Native(vp) end)
        if ok and native and native ~= '' then
          local h = _stripToUserHome(native)
          if h then return h end
        end
      end
      if FS.getFileRealPath then
        local ok, native = pcall(function() return FS:getFileRealPath(vp) end)
        if ok and native and native ~= '' then
          local h = _stripToUserHome(native)
          if h then return h end
        end
      end
    end
  end
  return nil
end

local function gvdDocsDir()
  if gvdDocsResolved then return gvdDocsResolved end
  local home = _tryHomeEnv() or _tryFsHome()
  local dir
  if home and home ~= '' then
    dir = home:gsub('\\', '/') .. '/Documents/GVD'
  else
    -- Last resort: still under Documents/GVD relative to CWD (never bare filename in userfolder root)
    dir = 'Documents/GVD'
  end
  -- Best-effort mkdir
  if FS and FS.directoryCreate then
    pcall(function() FS:directoryCreate(dir, true) end)
  end
  gvdDocsResolved = dir
  if not gvdDocsLogged then
    gvdDocsLogged = true
    log('I', 'GVD', '[GVD] docs dir: ' .. tostring(dir) .. ' (USERPROFILE=' .. tostring(os.getenv('USERPROFILE') or '') .. ')')
    print('[GVD] docs dir: ' .. tostring(dir))
  end
  return dir
end

local function gvdFile(name)
  return gvdDocsDir() .. '/' .. name
end

local function userEngagePath()
  return gvdFile('gvd_engage.json')
end

local function userCmdPath()
  return gvdFile('gvd_cmd.json')
end

local function userUiPrefsPath()
  return gvdFile('gvd_ui_prefs.json')
end

local function userStatePath()
  return gvdFile('gvd_state.json')
end


local function writeText(path, data)
  if not path then return false end
  -- Ensure parent Documents/GVD exists when FS can
  local parent = tostring(path):match('^(.+)/[^/]+$')
  if parent and FS and FS.directoryCreate then
    pcall(function() FS:directoryCreate(parent, true) end)
  end
  if FS and FS.writeFile then
    local ok = pcall(function() FS:writeFile(path, data) end)
    if ok then return true end
  end
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

local function getPlayerVeh()
  if be and be.getPlayerVehicle then
    return be:getPlayerVehicle(0)
  end
  return nil
end


local function encodeUiStateMirror(st)
  if jsonEncode then
    local ok, s = pcall(jsonEncode, st)
    if ok and s then return s end
  end
  if util_jsonEncode then
    local ok, s = pcall(util_jsonEncode, st)
    if ok and s then return s end
  end
  return nil
end

local function sanitizePolicy(p)
  p = tostring(p or ''):lower()
  if p == 'modular' or p == 'e2e' or p == 'shadow' then return p end
  return nil
end

local function sanitizeScreen(s)
  s = tostring(s or ''):lower()
  if s == 'auto' or s == '1' or s == '2' or s == '3' then return s end
  return nil
end

local function writeUiPrefs()
  -- policy / viz_screen are session requests: written only while the player asks for one this run,
  -- so a pref left over from a past session never overrides the supervisor's --policy at launch.
  local parts = {
    string.format('"show_path":%s', showPath and 'true' or 'false'),
    string.format('"show_agent_ghosts":%s', showAgentGhosts and 'true' or 'false'),
    string.format('"show_scene":%s', showScene and 'true' or 'false'),
  }
  if policyReq then parts[#parts + 1] = string.format('"policy":"%s"', policyReq) end
  if vizScreenReq then parts[#parts + 1] = string.format('"viz_screen":"%s"', vizScreenReq) end
  parts[#parts + 1] = string.format('"mtime":%d', os.time())
  writeText(userUiPrefsPath(), '{' .. table.concat(parts, ',') .. '}')
  -- Mirror into gvd_state so OpenCV / Python follow the same toggles
  local raw = readText(userStatePath())
  local st = raw and decodeJson(raw) or nil
  if type(st) == 'table' then
    st.gvd_show_path = showPath
    st.show_agent_ghosts = showAgentGhosts
    local encoded = encodeUiStateMirror(st)
    if encoded then
      writeText(userStatePath(), encoded)
    end
  end
end

local function readUiPrefs()
  local raw = readText(userUiPrefsPath())
  if not raw then return end
  local p = decodeJson(raw)
  if not p then return end
  if p.show_path ~= nil then showPath = not not p.show_path end
  if p.show_agent_ghosts ~= nil then showAgentGhosts = not not p.show_agent_ghosts end
  if p.show_scene ~= nil then showScene = not not p.show_scene end
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

local function clamp(x, a, b)
  if x < a then return a end
  if x > b then return b end
  return x
end

local function nowUnix()
  -- Must match Python time.time() / heartbeat_unix (NOT os.clock)
  return os.time()
end

-- 1:1 mapping: path_ego x=right, y=forward, z=up in vehicle frame → world via veh basis.
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
  -- World ribbon on pavement (not a 2D HUD). prism fail → 3 parallel lines L/C/R.
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
      -- 3 parallel lines so a thin center line is never the only fallback
      pcall(function() d:drawLine(af, bf, col) end)
      -- left/right offsets along a crude right vector
      local dx = (b.x or 0) - (a.x or 0)
      local dy = (b.y or 0) - (a.y or 0)
      local len = math.sqrt(dx * dx + dy * dy) + 1e-6
      local rx, ry = -dy / len * (width * 0.45), dx / len * (width * 0.45)
      local aL = vec3((a.x or 0) + rx, (a.y or 0) + ry, (a.z or 0))
      local bL = vec3((b.x or 0) + rx, (b.y or 0) + ry, (b.z or 0))
      local aR = vec3((a.x or 0) - rx, (a.y or 0) - ry, (a.z or 0))
      local bR = vec3((b.x or 0) - rx, (b.y or 0) - ry, (b.z or 0))
      local aLf = aL.toFloat3 and aL:toFloat3() or aL
      local bLf = bL.toFloat3 and bL:toFloat3() or bL
      local aRf = aR.toFloat3 and aR:toFloat3() or aR
      local bRf = bR.toFloat3 and bR:toFloat3() or bR
      pcall(function() d:drawLine(aLf, bLf, col) end)
      pcall(function() d:drawLine(aRf, bRf, col) end)
    end
  end
end

local function egoBasis(veh)
  local pos = veh:getPosition()
  local fwd = veh:getDirectionVector():normalized()
  local up = (veh:getDirectionVectorUp() or vec3(0, 0, 1)):normalized()
  local right = fwd:cross(up)
  if right:length() < 1e-6 then right = vec3(1, 0, 0) else right = right:normalized() end
  return pos, right, fwd, up
end

local function drawTrackHull(veh, tr, col)
  -- Vehicle ~4.2x1.8 / ped 0.6x0.6 in ego frame → world via same basis as path_ego (x right, y forward, z up)
  if not veh or not tr then return end
  local d = drawer()
  if not d then return end
  local pos, right, fwd, up = egoBasis(veh)
  local x = tonumber(tr.x) or 0
  local y = tonumber(tr.y) or 0
  local yaw = tonumber(tr.yaw or tr.heading) or 1.57
  local cls = tostring(tr['class'] or tr.class or 'vehicle')
  local L, W = 4.2, 1.8
  if cls == 'pedestrian' or cls == 'ped' then L, W = 0.6, 0.6 end
  if cls == 'bicycle' or cls == 'bike' then L, W = 1.8, 0.6 end
  local cy, sy = math.cos(yaw), math.sin(yaw)
  -- Tracker writes yaw as the ego-frame angle from +X, so pi/2 is straight ahead
  -- (python/perception/detect.py defaults to 1.57). Heading is (cos, sin), not (sin, cos).
  local rearX, rearY = x - (L * 0.5) * cy, y - (L * 0.5) * sy
  local frontX, frontY = x + (L * 0.5) * cy, y + (L * 0.5) * sy
  local pA = pos + right * rearX + fwd * rearY + up * Z_BIAS
  local pB = pos + right * frontX + fwd * frontY + up * Z_BIAS
  local aF = pA.toFloat3 and pA:toFloat3() or pA
  local bF = pB.toFloat3 and pB:toFloat3() or pB
  local half = float3 and float3(0.2, W, 0.2) or nil
  local drew = false
  if half and d.drawSquarePrism then
    local ok = pcall(function() d:drawSquarePrism(aF, bF, half, half, col) end)
    drew = ok
  end
  if not drew and d.drawLine then
    -- 4 footprint lines + short vertical edge (half width along the heading normal)
    local hx, hy = (W * 0.5) * sy, -(W * 0.5) * cy
    local corners = {
      {rearX - hx, rearY - hy}, {rearX + hx, rearY + hy},
      {frontX + hx, frontY + hy}, {frontX - hx, frontY - hy},
    }
    local wpts = {}
    for i = 1, 4 do
      local wx = corners[i][1]
      local wy = corners[i][2]
      wpts[i] = pos + right * wx + fwd * wy + up * Z_BIAS
    end
    for i = 1, 4 do
      local a = wpts[i]
      local b = wpts[(i % 4) + 1]
      local af = a.toFloat3 and a:toFloat3() or a
      local bf = b.toFloat3 and b:toFloat3() or b
      pcall(function() d:drawLine(af, bf, col) end)
    end
    local top = wpts[1] + up * 0.4
    local tF = top.toFloat3 and top:toFloat3() or top
    local b0 = wpts[1].toFloat3 and wpts[1]:toFloat3() or wpts[1]
    pcall(function() d:drawLine(b0, tF, col) end)
  end
end

local function drawTrackGhosts(st, veh, baseA)
  if not veh or not st then return end
  local tracks = st.tracks
  if not tracks then return end
  -- Honor state flag directly when set; else fall back to extension local / tracks_n>0
  local show
  if st.show_agent_ghosts ~= nil then
    show = not not st.show_agent_ghosts
  else
    local tn = tonumber(st.tracks_n) or #tracks
    show = showAgentGhosts or (tn > 0)
  end
  if not show then return end
  local cipv = nil
  if st.planner and st.planner.cipv_id ~= nil then cipv = tonumber(st.planner.cipv_id) end
  local count = 0
  for _, tr in ipairs(tracks) do
    if count >= MAX_TRACK_GHOSTS then break end
    local id = tonumber(tr.id)
    local isCipv = cipv and id and id == cipv
    local a = isCipv and math.floor(clamp(baseA, 0, 255)) or math.floor(clamp(baseA * 0.40, 0, 255))
    local col = color(ICE_R, ICE_G, ICE_B, a)
    drawTrackHull(veh, tr, col)
    count = count + 1
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
local hbCoarse = false   -- true when only heartbeat_unix (1 s resolution) is available
local hbKnown = false

local function noteHeartbeat(st)
  -- High-res path: Python heartbeat_mtime; age via os.clock since the beat last advanced.
  -- Fallback: heartbeat_unix (1 s resolution) → extra slack.
  if not st then return end
  local mt = tonumber(st.heartbeat_mtime)
  local coarse = false
  if mt == nil then
    mt = tonumber(st.heartbeat_unix)
    coarse = true
  end
  if mt == nil then return end
  hbKnown = true
  hbCoarse = coarse
  if lastBeatMtime == nil or mt > lastBeatMtime + 1e-6 or mt < lastBeatMtime - 2.0 then
    -- forward = new beat; far backwards = the file was replaced (restart / restored old state),
    -- so trust the file over our own history and let the wall-clock check below judge it.
    lastBeatMtime = mt
    lastBeatMono = os.clock()
  end
end

local function hbLimitS()
  return hbCoarse and (1 + HB_STALE_S) or HB_STALE_S
end

local function hbAgeS()
  -- nil = no heartbeat field at all (treat as alive, matches pre-M5 states)
  if not hbKnown or lastBeatMono == nil then return nil end
  local mono = os.clock() - lastBeatMono
  -- A state file left over from a past session must read stale on the first poll, not fresh.
  if lastBeatMtime and lastBeatMtime > 1e9 then
    local wall = nowUnix() - lastBeatMtime
    if wall > 2.0 then return wall end
  end
  if mono < 0 then return 0 end
  return mono
end

local function heartbeatAlive(st, dt)
  if not st then return true end
  noteHeartbeat(st)
  local age = hbAgeS()
  if age == nil then return true end
  local alive = age <= hbLimitS()
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
  width = clamp(width, 1.8, 2.4)
  if conf < 0.45 then
    width = width * 0.75
    local keep = math.max(8, math.floor(#pts * 0.55))
    while #pts > keep do table.remove(pts) end
  end

  local a = fadeAlpha(isPreview and 90 or 200, conf, hbAlive)
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
  drawTrackGhosts(lastGood, veh, a)

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

local function r2(v)
  local n = tonumber(v)
  if n == nil then return nil end
  return math.floor(n * 100 + 0.5) / 100
end

local function uiPathPoints(st)
  local src = st and st.path_ego
  if type(src) ~= 'table' or #src < 2 then return nil end
  local step = math.max(1, math.floor(#src / UI_PATH_PTS))
  local out = {}
  for i = 1, #src, step do
    local p = src[i]
    out[#out + 1] = { x = r2(p.x or p[1] or 0), y = r2(p.y or p[2] or 0) }
    if #out >= UI_PATH_PTS then break end
  end
  if #out < 2 then return nil end
  return out
end

local function uiTracks(st)
  local src = st and st.tracks
  if type(src) ~= 'table' or #src == 0 then return nil end
  local cipv = nil
  if st.planner and st.planner.cipv_id ~= nil then cipv = tonumber(st.planner.cipv_id) end
  local out = {}
  for _, tr in ipairs(src) do
    if #out >= UI_TRACKS then break end
    local id = tonumber(tr.id)
    out[#out + 1] = {
      id = id,
      cls = tostring(tr['class'] or 'vehicle'),
      x = r2(tr.x or 0),
      y = r2(tr.y or 0),
      yaw = r2(tr.yaw or tr.heading or 1.57),
      v = r2(tr.speed_mps or 0),
      lead = (cipv ~= nil and id ~= nil and id == cipv) or false,
    }
  end
  return out
end

local function uiLanes(st)
  local src = st and st.lanes_bev
  if type(src) ~= 'table' then return nil end
  local out = {}
  for _, poly in ipairs(src) do
    if #out >= UI_LANES then break end
    if type(poly) == 'table' and #poly > 1 then
      local step = math.max(1, math.floor(#poly / UI_LANE_PTS))
      local pts = {}
      for i = 1, #poly, step do
        local p = poly[i]
        pts[#pts + 1] = { x = r2(p.x or p[1] or 0), y = r2(p.y or p[2] or 0) }
        if #pts >= UI_LANE_PTS then break end
      end
      if #pts > 1 then out[#out + 1] = pts end
    end
  end
  if #out == 0 then return nil end
  return out
end

local function uiCams(st)
  local ch = st and st.cam_health
  local ok, total = 0, 0
  local list = {}
  if type(ch) == 'table' then
    for _, id in ipairs(CAM_IDS) do
      local v = ch[id]
      if v ~= nil then
        total = total + 1
        local s = tostring(v)
        if s == 'ok' then ok = ok + 1 end
        list[#list + 1] = { id = id, h = s }
      end
    end
  end
  return ok, total, list
end

local function linkState()
  if not lastGood then return 'none' end
  local age = hbAgeS()
  if age == nil then return 'live' end
  if age <= hbLimitS() then return 'live' end
  return 'stale'
end

local function uiPayload()
  local st = lastGood
  local pl = (st and st.planner) or {}
  local ego = (st and st.ego) or {}
  local link = linkState()
  local mode = tostring((st and st.policy) or 'modular')
  local hz = tonumber(st and st.loop_hz) or 0
  local ttc = pl.ttc_lead
  local ttcS = (ttc == nil) and '--' or string.format('%.1f', tonumber(ttc) or 0)
  local n = tonumber((st and (st.tracks_n or st.objects_n))) or 0
  local line
  if st then
    line = string.format('GVD  %s|%s  %.0fHz  TTC %s  N=%d', mode, engaged and 'ON' or 'OFF', hz, ttcS, n)
  else
    line = 'GVD  ' .. (engaged and 'ON' or 'OFF') .. '  no telemetry'
  end
  local camOk, camTotal, camList = uiCams(st)

  return {
    -- engage / safety
    engaged = engaged,
    link = link,                                   -- live | stale | none
    hbAge = r2(hbAgeS()),
    disengageReason = st and tostring(st.disengage_reason or 'none') or nil,
    -- in-world viz toggles
    showPath = showPath,
    showGhosts = showAgentGhosts,
    showScene = showScene,
    -- policy (honest toys)
    policy = st and mode or nil,
    policyReq = policyReq,
    e2eOk = st and (st.e2e_ok ~= false) or nil,
    vetoReason = st and tostring(st.veto_reason or 'none') or nil,
    e2eBackend = st and st.e2e_backend or nil,
    -- driving numbers
    hz = hz,
    camHz = r2(st and st.camera_hz),
    ttc = (ttc == nil) and nil or r2(ttc),
    aeb = pl.aeb and tostring(pl.aeb) or nil,
    n = n,
    speed = r2(ego.speed_mps),
    steerDeg = r2(ego.steer_deg),
    pathConf = r2(st and st.path_conf),
    pathWidth = r2(st and st.path_width),
    pathPreview = st and (st.path_debug_preview ~= false) or nil,
    laneConf = r2(st and st.lane_conf),
    -- cameras (retail honesty: main only stays main only)
    camBackend = st and st.capture_backend or nil,
    camNote = st and st.capture_note or nil,
    camOk = st and camOk or nil,
    camTotal = st and camTotal or nil,
    cams = st and camList or nil,
    -- GVD VISION window (Python OpenCV second screen)
    vizWindow = st and st.viz_window or nil,
    vizScreen = st and st.viz_screen or nil,
    vizNote = st and st.viz_note or nil,
    vizScreenReq = vizScreenReq,
    -- nerd-light
    inferMs = r2(st and st.infer_ms),
    rssMb = r2(st and st.rss_mb),
    vramUsed = r2(st and st.gpu_vram_used_gb),
    vramTotal = r2(st and st.gpu_vram_total_gb),
    gpu = st and st.gpu_name or nil,
    detector = st and st.detector or nil,
    actuator = st and st.actuator or nil,
    cmdApplied = st and st.cmd_applied or nil,
    cmdReason = st and st.cmd_reason or nil,
    clipTrigger = st and st.last_clip_trigger or nil,
    encodeBackend = st and st.encode_backend or nil,
    -- scene geometry (ego frame: x right, y forward), capped + rounded
    path = (showScene and showPath) and uiPathPoints(st) or nil,
    tracks = (showScene and showAgentGhosts) and uiTracks(st) or nil,
    lanes = showScene and uiLanes(st) or nil,
    mode = mode .. (engaged and '|ON' or '|OFF'),
    text = line,
  }
end

local function pushUi()
  local p = uiPayload()
  if guihooks and guihooks.trigger then
    pcall(function()
      guihooks.trigger('gvdStrip', { text = p.text, mode = p.mode, hz = p.hz, ttc = p.ttc, n = p.n, engaged = engaged })
      guihooks.trigger('gvdUi', p)
    end)
  end

  if not lastGood then return end
  -- Screen-adjacent draw if API exists (compact; not a nerd panel)
  local d = drawer()
  local veh = getPlayerVeh()
  if d and d.drawTextAdvanced and veh then
    local pos = veh:getPosition()
    local up = (veh:getDirectionVectorUp() or vec3(0, 0, 1)):normalized()
    local anchor = pos + up * 2.2
    local af = anchor.toFloat3 and anchor:toFloat3() or anchor
    pcall(function()
      d:drawTextAdvanced(af, p.text, color(200, 204, 212, 200), true, false, color(12, 13, 16, 140))
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
  noteHeartbeat(st)
  -- UI prefs win: if gvd_ui_prefs.json exists, do NOT apply path/ghosts from gvd_state
  -- (run_vision show_agent_ghosts=true would clobber "Show ghosts" off every tick).
  local prefsRaw = readText(userUiPrefsPath())
  if prefsRaw and prefsRaw ~= '' then
    readUiPrefs()
  else
    if st.gvd_show_path ~= nil then showPath = not not st.gvd_show_path end
    if st.show_agent_ghosts ~= nil then showAgentGhosts = not not st.show_agent_ghosts end
  end
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

local preRenderSeen = false

local function tickPush(dt)
  stripAcc = stripAcc + (dt or 0)
  -- 10 Hz while the app draws the VISION scene, 4 Hz for the plain status strip
  local every = (lastGood and showScene) and 0.10 or 0.25
  if stripAcc >= every then
    stripAcc = 0
    pushUi()
  end
end

function M.onPreRender(dt)
  preRenderSeen = true
  pollState(dt)
  M.drawPath(dt)
  tickPush(dt)
end

function M.onDebugDraw(_focuspos)
  -- Also draw here if onPreRender is not hooked in this build.
  -- Alt+A live ribbon remains UNPROVEN on Linux.
  M.drawPath(0.016)
end

function M.onUpdate(dt)
  pollState(dt)
  applyCmdJson(dt)
  -- Builds that never call onPreRender would otherwise leave the app with no data.
  if not preRenderSeen then tickPush(dt) end
end

function M.onExtensionLoaded()
  gvdDocsDir()  -- resolve + log once
  readUiPrefs()
  -- Soft: keyboard.diff / actions load race — reload bindings after gvd actions are present
  pcall(function()
    if core_input_bindings and core_input_bindings.reloadBindings then
      core_input_bindings.reloadBindings()
    elseif extensions and extensions.core_input_bindings and extensions.core_input_bindings.reloadBindings then
      extensions.core_input_bindings.reloadBindings()
    end
  end)
  log('I', 'GVD', '[GVD] loaded. Alt+A engage. Path: GVD PATH. Strip: mode/Hz/TTC/N. UI app: GVD.')
  print('[GVD] loaded. Alt+A engage. Writes ' .. userEngagePath() .. '; optional gvd_cmd.json poll (BeamNGpy preferred).')
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
  pushUi()
end

function M.setShowAgentGhosts(v)
  showAgentGhosts = not not v
  writeUiPrefs()
  pushUi()
end

function M.setShowPath(v)
  showPath = not not v
  writeUiPrefs()
  pushUi()
end

function M.setShowScene(v)
  showScene = not not v
  writeUiPrefs()
  pushUi()
end

-- Session request for the Python supervisor: modular | e2e | shadow.
-- The modular safety supervisor still vetoes E2E; this only picks which intent may be applied.
function M.requestPolicy(p)
  local want = sanitizePolicy(p)
  if not want then return end
  policyReq = want
  writeUiPrefs()
  log('I', 'GVD', '[GVD] policy request: ' .. want .. ' (applies while the Python supervisor is running)')
  pushUi()
end

-- Session request: which monitor the Python GVD VISION window should sit on (auto|1|2|3).
function M.requestVizScreen(s)
  local want = sanitizeScreen(s)
  if not want then return end
  vizScreenReq = want
  writeUiPrefs()
  log('I', 'GVD', '[GVD] GVD VISION screen request: ' .. want)
  pushUi()
end

function M.pushUiState()
  pushUi()
end

function M.setEngaged(v)
  local want = not not v
  if engaged ~= want then
    M.toggleEngage()
  end
end

function M.engage()
  if not engaged then M.toggleEngage() end
end

function M.disengage()
  if engaged then M.toggleEngage() end
end

function M.isEngaged()
  return engaged
end

function M.getShowPath()
  return showPath
end

function M.getShowAgentGhosts()
  return showAgentGhosts
end

function M.getShowScene()
  return showScene
end

M.onInit = onInit

return M
