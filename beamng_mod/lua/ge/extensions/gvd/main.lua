-- Grok Vision Drive — GE extension: engage + ice-blue ego path + compact HUD strip + retail drive
-- NOTE: Alt+A live ribbon/drive remain UNPROVEN on Linux; confirm on Windows BeamNG smoke.
-- Retail (window capture): while engaged, Documents/GVD/gvd_cmd.json is applied to the player vehicle
-- through vehicle-Lua input.event (the calls BeamNG's AI / BeamNGpy use). No DLL / hooks / injection.
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
local EGO_REL = 'Documents/GVD/gvd_ego.json'
local lastCmdSeq = -1
local cmdAcc = 0
local pollAcc = 0
local pollEvery = 0.10
local fadeAcc = 0
local fadeDur = 0.40
local lastGood = nil
local missingKeysLogged = false
local stripAcc = 0

-- M6 retail drive state (gvd_cmd.json → player vehicle; electrics echo → gvd_ego.json)
local CMD_POLL_S = 0.05      -- 20 Hz apply
-- Dead-man timers (wall / seq age): STALE = brake hold; DEAD = release + disengage.
-- STALE matches Python HEARTBEAT_STALE_S (0.35). DEAD is short (1.0s) so a dead
-- supervisor doesn't leave the car braked indefinitely — was 3.0s, soft-tightened.
local CMD_STALE_S = 0.35     -- no new seq for this long → brake hold (dead-man)
local CMD_DEAD_S = 1.0       -- stream dead this long while we hold the car → release + disengage
local EGO_POLL_S = 0.10      -- 10 Hz electrics echo while the supervisor is alive
local applying = false       -- true while our input.event stream holds the player vehicle
local applyVeh = nil         -- vehicle object we last applied to (released on switch/disengage)
local lastAppliedSeq = -1
local cmdStaleAcc = 0
local cmdStaleLogged = false
local arcadeQueued = false
local egoAcc = 0
local egoFb = nil
local stateBeatAcc = 0       -- seconds since gvd_state.json heartbeat_mtime last changed
local lastStateBeat = nil

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
local UI_LANES = 8
local UI_LANE_PTS = 12
local UI_EDGES = 2
local UI_SIGNS = 8
local UI_FANS = 6
local UI_FAN_PTS = 8
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

local function userEgoPath()
  return gvdFile('gvd_ego.json')
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

local lastEngageWriteUnix = 0
local luaDisengageReason = nil   -- why *we* switched off, so the HUD can say so before the next state read

local function writeEngageFile(reason)
  lastEngageWriteUnix = os.time()
  luaDisengageReason = (not engaged) and reason or nil
  local payload = string.format('{"engaged":%s,"mtime":%d,"disengage_reason":"%s"}',
    engaged and 'true' or 'false', lastEngageWriteUnix, tostring(reason or 'none'))
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

local function uiPoly(poly, cap)
  if type(poly) ~= 'table' or #poly < 2 then return nil end
  local step = math.max(1, math.floor(#poly / cap))
  local pts = {}
  for i = 1, #poly, step do
    local p = poly[i]
    pts[#pts + 1] = { x = r2(p.x or p[1] or 0), y = r2(p.y or p[2] or 0) }
    if #pts >= cap then break end
  end
  if #pts < 2 then return nil end
  return pts
end

-- lanes_ext carries kind detected|predicted|stub per boundary; lanes_bev (older supervisor)
-- only ever holds what the Hough fit actually saw, so it maps to kind=detected.
local function uiLanes(st)
  if type(st) ~= 'table' then return nil end
  local out = {}
  local ext = st.lanes_ext
  if type(ext) == 'table' and #ext > 0 then
    for _, ln in ipairs(ext) do
      if #out >= UI_LANES then break end
      local pts = uiPoly(ln.points, UI_LANE_PTS)
      if pts then
        out[#out + 1] = {
          pts = pts,
          kind = tostring(ln.kind or 'detected'),
          side = tostring(ln.side or ''),
          style = tostring(ln.style or 'unknown'),
          idx = tonumber(ln.index) or 0,
        }
      end
    end
  elseif type(st.lanes_bev) == 'table' then
    for _, poly in ipairs(st.lanes_bev) do
      if #out >= UI_LANES then break end
      local pts = uiPoly(poly, UI_LANE_PTS)
      if pts then
        out[#out + 1] = { pts = pts, kind = 'detected', side = '', style = 'unknown', idx = 0 }
      end
    end
  end
  if #out == 0 then return nil end
  return out
end

local function uiEdges(st)
  local src = st and st.road_edges
  if type(src) ~= 'table' then return nil end
  local out = {}
  for _, e in ipairs(src) do
    if #out >= UI_EDGES then break end
    local pts = uiPoly(e.points, UI_LANE_PTS)
    if pts then
      out[#out + 1] = { pts = pts, kind = tostring(e.kind or 'predicted'), side = tostring(e.side or '') }
    end
  end
  if #out == 0 then return nil end
  return out
end

local function uiSigns(st)
  local src = st and st.signs
  if type(src) ~= 'table' or #src == 0 then return nil end
  local out = {}
  for _, s in ipairs(src) do
    if #out >= UI_SIGNS then break end
    out[#out + 1] = {
      cls = tostring(s.cls or 'sign'),
      x = r2(s.x or 0),
      y = r2(s.y or 0),
      conf = r2(s.conf),
      state = s.state and tostring(s.state) or nil,
    }
  end
  return out
end

-- Forecast fans: mode-0 constant-yaw-rate toy written by the supervisor (state.agents).
local function uiFans(st)
  local src = st and st.agents
  if type(src) ~= 'table' or #src == 0 then return nil end
  local out = {}
  for _, ag in ipairs(src) do
    if #out >= UI_FANS then break end
    local pts = uiPoly(ag.path_ego, UI_FAN_PTS)
    if pts then out[#out + 1] = { id = tonumber(ag.id), pts = pts } end
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

-- `st and st.flag or nil` would turn a real false into nil, so booleans go through here.
local function sBool(st, v, dflt)
  if not st then return nil end
  if v == nil then return dflt end
  return not not v
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
  local modeTag = engaged and (applying and '|DRIVE' or '|ON') or '|OFF'
  local line
  if st then
    line = string.format('GVD  %s%s  %.0fHz  TTC %s  N=%d', mode, modeTag, hz, ttcS, n)
  else
    line = 'GVD  ' .. (engaged and 'ON' or 'OFF') .. '  no telemetry'
  end
  local camOk, camTotal, camList = uiCams(st)

  return {
    -- engage / safety
    engaged = engaged,
    applying = applying,                           -- M6: our input.event stream holds the player vehicle
    link = link,                                   -- live | stale | none
    hbAge = r2(hbAgeS()),
    disengageReason = luaDisengageReason or (st and tostring(st.disengage_reason or 'none')) or nil,
    -- in-world viz toggles
    showPath = showPath,
    showGhosts = showAgentGhosts,
    showScene = showScene,
    -- policy (honest toys)
    policy = st and mode or nil,
    policyReq = policyReq,
    e2eOk = sBool(st, st and st.e2e_ok, true),
    vetoReason = st and tostring(st.veto_reason or 'none') or nil,
    e2eBackend = st and st.e2e_backend or nil,
    -- driving numbers
    hz = hz,
    camHz = r2(st and st.camera_hz),
    ttc = (ttc == nil) and nil or r2(ttc),
    aeb = pl.aeb and tostring(pl.aeb) or nil,
    n = n,
    speed = r2(ego.speed_mps),
    targetV = r2(pl.target_v),          -- app draws slow-down chevrons off these two
    brakeCmd = r2(ego.brake),
    steerDeg = r2(ego.steer_deg),
    pathConf = r2(st and st.path_conf),
    pathWidth = r2(st and st.path_width),
    pathPreview = sBool(st, st and st.path_debug_preview, true),
    laneConf = r2(st and st.lane_conf),
    -- cameras (retail honesty: main only stays main only)
    camBackend = st and st.capture_backend or nil,
    camNote = st and st.capture_note or nil,
    camOk = st and camOk or nil,
    camTotal = st and camTotal or nil,
    cams = st and camList or nil,
    -- GVD VISION window (Python OpenCV second screen)
    vizWindow = sBool(st, st and st.viz_window, false),
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
    cmdApplied = sBool(st, st and st.cmd_applied, false),
    cmdReason = st and st.cmd_reason or nil,
    cmdSeq = st and tonumber(st.cmd_seq) or nil,
    cmdAckSeq = st and tonumber(st.cmd_ack_seq) or nil,
    egoSource = st and st.ego_source or nil,
    clipTrigger = st and st.last_clip_trigger or nil,
    encodeBackend = st and st.encode_backend or nil,
    -- scene geometry (ego frame: x right, y forward), capped + rounded
    path = (showScene and showPath) and uiPathPoints(st) or nil,
    tracks = (showScene and showAgentGhosts) and uiTracks(st) or nil,
    fans = (showScene and showAgentGhosts) and uiFans(st) or nil,
    lanes = showScene and uiLanes(st) or nil,
    edges = showScene and uiEdges(st) or nil,
    signs = showScene and uiSigns(st) or nil,
    mode = mode .. modeTag,
    text = line,
  }
end

local function pushUi()
  local p = uiPayload()
  if guihooks and guihooks.trigger then
    pcall(function()
      guihooks.trigger('gvdStrip', {
        text = p.text, mode = p.mode, hz = p.hz, ttc = p.ttc, n = p.n,
        engaged = engaged, applying = applying,
      })
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

-- ───────────── M6 retail drive: gvd_cmd.json → player vehicle, electrics → gvd_ego.json ─────────────
-- Python writes {steer,throttle,brake,seq,engaged,heartbeat_mtime} every tick. While Lua-engaged AND the
-- payload says engaged AND the seq keeps advancing, we feed the player vehicle with the vehicle-Lua calls
-- BeamNG's own AI / BeamNGpy use: input.event('steering', v, 1) (pad-smoothed, +1 = right like kbdSteer)
-- and input.event('throttle'|'brake', v, 2) (direct). Vehicle Lua echoes electrics back through
-- obj:queueGameEngineLua → M.onEgoFeedback → gvd_ego.json (wheelspeed, inputs, applied seq).
local VE_APPLY_FMT = "input.event('steering',%.4f,1);input.event('throttle',%.4f,2);input.event('brake',%.4f,2)"
local VE_RELEASE = "input.event('steering',0,1);input.event('throttle',0,2);input.event('brake',0,2)"
local VE_ARCADE = "if drivetrain and drivetrain.setShifterMode then pcall(drivetrain.setShifterMode,'arcade') end"
local VE_FEEDBACK = "local ev=(electrics and electrics.values) or {};"
  .. "local function n(x) x=tonumber(x) or 0;if x~=x or x==math.huge or x==-math.huge then x=0 end;return x end;"
  .. "obj:queueGameEngineLua(string.format('extensions.gvd_main.onEgoFeedback(%.3f,%.4f,%.4f,%.4f)',"
  .. "n(ev.wheelspeed or ev.airspeed),n(ev.steering_input),n(ev.throttle_input),n(ev.brake_input)))"

-- ───────────── driver override: force-feedback deadband on steer, hard pedals ─────────────
-- Force-feedback / racing wheels move electrics.steering_input around whatever GVD commands
-- (self-aligning torque, spring centering, kicks over bumps), so a bare threshold on the echo
-- read that chatter as a driver and accidentally disengaged GVD. Steer now goes through
--   residual (echo - envelope of what we commanded) → deadband → hysteresis → dwell
-- so the trip point sits at steer_enter + steer_deadband of lock and the residual has to hold
-- one direction for steer_hold_s to count. Pedals stay hard: a press past its threshold is an
-- override on the tick it lands.
-- Same maths as python/control/override.py; these defaults must match config/control.yaml,
-- which the supervisor mirrors into gvd_state.json.override_cfg.
local OVR = {
  steer_deadband = 0.10,
  steer_enter = 0.55,
  steer_clear = 0.30,
  steer_hold_s = 0.12,
  steer_cmd_max = 0.85,
  steer_sign_flip_resets = true,
  brake_enter = 0.08,
  throttle_enter = 0.15,
  pedal_hold_s = 0.0,
  cmd_window_s = 0.30,
  ffb_assume_wheel = true,
}
local OVR_RING = 64
local OVR_ECHO_MAX_S = 0.5   -- a frozen electrics echo is the dead-man's problem, not an override
local ovrCmds = {}           -- {t, s, th, b} we actually pushed to the car
local ovrClock = 0           -- monotonic seconds, accumulated from the applyCmdJson step
local ovrEchoStamp = nil     -- ovrClock when the last electrics echo landed
local ovrArmedAt = nil       -- ovrClock of the first command since we took the car
local ovrSteerHeld = 0
local ovrSteerSign = 0
local ovrPedalHeld = 0
local ovrCfgLogged = false

local function ovrReset()
  ovrCmds = {}
  ovrArmedAt = nil
  ovrSteerHeld = 0
  ovrSteerSign = 0
  ovrPedalHeld = 0
end

local function ovrNoteCommand(steer, throttle, brake)
  if ovrArmedAt == nil then ovrArmedAt = ovrClock end
  ovrCmds[#ovrCmds + 1] = { t = ovrClock, s = steer, th = throttle, b = brake }
  -- Keep one sample past the window: an empty envelope would read every echo as a driver.
  while #ovrCmds > 1 and (ovrClock - ovrCmds[1].t) > OVR.cmd_window_s do table.remove(ovrCmds, 1) end
  while #ovrCmds > OVR_RING do table.remove(ovrCmds, 1) end
end

local function ovrEnvelope(key)
  local lo, hi
  for _, c in ipairs(ovrCmds) do
    local v = c[key]
    if lo == nil or v < lo then lo = v end
    if hi == nil or v > hi then hi = v end
  end
  return lo or 0, hi or 0
end

-- Signed distance outside the command envelope: 0 while the echo is still explainable as ours.
local function ovrResidual(v, lo, hi)
  if v > hi then return v - hi end
  if v < lo then return v - lo end
  return 0
end

-- Subtractive deadband, the way a wheel axis dead zone works: inside the band there is no
-- driver, outside it the band comes off the residual, so steer_enter is a threshold on the
-- compensated signal and the trip point sits that much further out.
local function ovrDeadband(v, width)
  local mag = math.abs(v) - width
  if mag <= 0 then return 0 end
  return (v > 0) and mag or -mag
end

local function readOverrideCfg(st)
  local c = st and st.override_cfg
  if type(c) ~= 'table' then return end
  local function num(k, lo, hi)
    local v = tonumber(c[k])
    if v == nil or v ~= v then return end
    OVR[k] = clamp(v, lo, hi)
  end
  num('steer_deadband', 0, 0.5)
  num('steer_enter', 0.02, 1)
  num('steer_clear', 0, 1)
  num('steer_hold_s', 0, 2)
  num('steer_cmd_max', 0, 1)
  num('brake_enter', 0.01, 1)
  num('throttle_enter', 0.01, 1)
  num('pedal_hold_s', 0, 2)
  num('cmd_window_s', 0, 2)
  if c.steer_sign_flip_resets ~= nil then OVR.steer_sign_flip_resets = not not c.steer_sign_flip_resets end
  if c.ffb_assume_wheel ~= nil then OVR.ffb_assume_wheel = not not c.ffb_assume_wheel end
  -- Whatever the yaml says, the dwell has to be able to discharge.
  if OVR.steer_clear >= OVR.steer_enter then OVR.steer_clear = OVR.steer_enter * 0.95 end
  if not ovrCfgLogged then
    ovrCfgLogged = true
    log('I', 'GVD', string.format(
      '[GVD] override thresholds from gvd_state: steer deadband %.2f trip %.2f hold %.2fs, brake %.2f, throttle %.2f',
      OVR.steer_deadband, OVR.steer_enter, OVR.steer_hold_s, OVR.brake_enter, OVR.throttle_enter))
  end
end

-- GE Lua has no documented "is a force-feedback wheel attached" query, so this is best effort:
-- true only when an enumerable device names a wheel, false only when a device list came back
-- without one, nil (unknown) otherwise. Unknown keeps the deadband on. UNPROVEN on Windows.
local WHEEL_HINTS = { 'wheel', 'ffb', 'logitech', 'thrustmaster', 'fanatec', 'simucube', 'g27', 'g29', 'g920', 't300', 'csl' }
local ffbWheel = nil
local ffbProbed = false

local function looksLikeWheel(name)
  name = tostring(name or ''):lower()
  if name == '' then return false end
  for _, k in ipairs(WHEEL_HINTS) do
    if name:find(k, 1, true) then return true end
  end
  return false
end

local function probeFfbWheel()
  if ffbProbed then return ffbWheel end
  ffbProbed = true
  local names = nil
  local function collect(t)
    if type(t) ~= 'table' then return end
    for k, v in pairs(t) do
      local name = nil
      if type(v) == 'string' then name = v
      elseif type(v) == 'table' then name = v.name or v.deviceName or v.product end
      if name == nil and type(k) == 'string' then name = k end
      if name ~= nil then
        names = names or {}
        names[#names + 1] = tostring(name)
      end
    end
  end
  for _, src in ipairs({ 'core_input_bindings', 'WinInput', 'core_input_actionFilter' }) do
    local ok, mod = pcall(function() return _G[src] end)
    if ok and type(mod) == 'table' then
      for _, key in ipairs({ 'devices', 'deviceNames', 'inputDevices' }) do
        pcall(function() collect(mod[key]) end)
      end
    end
  end
  if names == nil then
    log('I', 'GVD', '[GVD] force-feedback wheel: undetectable on this build — steer deadband stays on while engaged')
    return nil
  end
  for _, n in ipairs(names) do
    if looksLikeWheel(n) then
      ffbWheel = true
      log('I', 'GVD', '[GVD] force-feedback wheel detected (' .. n .. '): steer deadband on')
      return true
    end
  end
  ffbWheel = false
  log('I', 'GVD', '[GVD] no wheel among ' .. #names .. ' input devices')
  return false
end

local function ovrSteerBand()
  local wheel = probeFfbWheel()  -- probed on the first drive tick so the log lands once
  if OVR.ffb_assume_wheel or wheel ~= false then return OVR.steer_deadband end
  return 0
end

-- 'none' | 'steer' | 'brake' | 'throttle'. Only meaningful while we hold the vehicle.
local function ovrCheck(dt)
  if not egoFb or ovrEchoStamp == nil then return 'none' end
  if (ovrClock - ovrEchoStamp) > OVR_ECHO_MAX_S then return 'none' end
  -- Warm-up: until we have commanded across the whole lookback window there is nothing to
  -- attribute the echo to, and a player holding the brake as they press Alt+A would override
  -- themselves on the spot.
  if ovrArmedAt == nil or (ovrClock - ovrArmedAt) < OVR.cmd_window_s then
    ovrSteerHeld = 0
    ovrSteerSign = 0
    ovrPedalHeld = 0
    return 'none'
  end

  local thrLo, thrHi = ovrEnvelope('th')
  local brkLo, brkHi = ovrEnvelope('b')
  local thrR = math.max(0, ovrResidual(clamp(egoFb.throttle or 0, 0, 1), thrLo, thrHi))
  local brkR = math.max(0, ovrResidual(clamp(egoFb.brake or 0, 0, 1), brkLo, brkHi))
  local pedal = 'none'
  if brkR >= OVR.brake_enter and brkR >= thrR then
    pedal = 'brake'
  elseif thrR >= OVR.throttle_enter then
    pedal = 'throttle'
  end
  if pedal == 'none' then ovrPedalHeld = 0 else ovrPedalHeld = ovrPedalHeld + dt end

  local sLo, sHi = ovrEnvelope('s')
  local steerR = 0
  if math.max(math.abs(sLo), math.abs(sHi)) <= OVR.steer_cmd_max then
    steerR = ovrDeadband(ovrResidual(clamp(egoFb.steer or 0, -1, 1), sLo, sHi), ovrSteerBand())
  end
  local mag = math.abs(steerR)
  local sign = (steerR > 0 and 1) or (steerR < 0 and -1) or 0
  if mag >= OVR.steer_enter then
    if OVR.steer_sign_flip_resets and ovrSteerSign ~= 0 and sign ~= ovrSteerSign then
      ovrSteerHeld = 0   -- chatter alternates side to side; a driver pulling the wheel does not
    end
    ovrSteerSign = sign
    ovrSteerHeld = ovrSteerHeld + dt
  elseif mag <= OVR.steer_clear then
    ovrSteerHeld = 0
    ovrSteerSign = 0
  end
  -- Between clear and enter the dwell is frozen: that band is the hysteresis.

  if pedal ~= 'none' and ovrPedalHeld >= OVR.pedal_hold_s then return pedal end
  if mag >= OVR.steer_enter and ovrSteerHeld >= OVR.steer_hold_s then return 'steer' end
  return 'none'
end

local function queueVehicle(veh, code)
  if not veh or not veh.queueLuaCommand then return false end
  local ok = pcall(function() veh:queueLuaCommand(code) end)
  return ok
end

local function applyInputs(veh, steer, throttle, brake)
  steer = clamp(tonumber(steer) or 0, -1, 1)
  throttle = clamp(tonumber(throttle) or 0, 0, 1)
  brake = clamp(tonumber(brake) or 0, 0, 1)
  if brake > 0.01 then throttle = 0 end  -- never both pedals (AEB semantics; arcade auto-reverse guard)
  ovrNoteCommand(steer, throttle, brake)  -- baseline the override residual reads against
  return queueVehicle(veh, string.format(VE_APPLY_FMT, steer, throttle, brake))
end

local function vehId(veh)
  if not veh then return nil end
  local ok, id = pcall(function() return veh:getID() end)
  if ok and id ~= nil then return id end
  return veh
end

local function releaseInputs(why)
  if not applying then return end
  applying = false
  queueVehicle(applyVeh or getPlayerVeh(), VE_RELEASE)
  applyVeh = nil
  ovrReset()  -- hands off: the command envelope and the override dwell mean nothing now
  log('I', 'GVD', '[GVD] released vehicle inputs (' .. tostring(why) .. ')')
end

local function supervisorAlive()
  return lastGood ~= nil and stateBeatAcc <= 2.0
end

local function writeEgoFile()
  if not egoFb then return end
  local payload = string.format(
    '{"speed_mps":%.3f,"steering_input":%.4f,"throttle_input":%.4f,"brake_input":%.4f,"applied_seq":%d,"applying":%s,"mtime":%d}',
    egoFb.speed, egoFb.steer, egoFb.throttle, egoFb.brake,
    math.floor(tonumber(lastAppliedSeq) or -1), applying and 'true' or 'false', os.time())
  writeText(userEgoPath(), payload)
end

-- Called from vehicle Lua (VE_FEEDBACK) via obj:queueGameEngineLua.
function M.onEgoFeedback(speed, steerIn, thrIn, brkIn)
  egoFb = {
    speed = tonumber(speed) or 0,
    steer = tonumber(steerIn) or 0,
    throttle = tonumber(thrIn) or 0,
    brake = tonumber(brkIn) or 0,
  }
  ovrEchoStamp = ovrClock
  writeEgoFile()
end

local function pollEgo(dt)
  egoAcc = egoAcc + (dt or 0)
  if egoAcc < EGO_POLL_S then return end
  egoAcc = 0
  if not supervisorAlive() then return end
  local veh = getPlayerVeh()
  if not veh then return end
  queueVehicle(veh, VE_FEEDBACK)
end

local function applyCmdJson(dt)
  cmdAcc = cmdAcc + (dt or 0)
  if cmdAcc < CMD_POLL_S then return end
  local step = cmdAcc
  cmdAcc = 0
  ovrClock = ovrClock + step
  if not engaged then
    releaseInputs('disengaged')
    cmdStaleAcc = 0
    return
  end
  local veh = getPlayerVeh()
  if not veh then
    applying = false  -- vehicle gone; nothing to release
    applyVeh = nil
    return
  end
  if applying and applyVeh and vehId(applyVeh) ~= vehId(veh) then
    -- Player switched vehicles while we held the wheel: free the old one, re-arm arcade for the new one.
    queueVehicle(applyVeh, VE_RELEASE)
    arcadeQueued = false
    -- New car, new baseline: its electrics echo says nothing about the old car's commands.
    ovrReset()
    ovrEchoStamp = nil
    log('I', 'GVD', '[GVD] player vehicle changed: released previous vehicle inputs')
  end
  applyVeh = veh
  local raw = readText(userCmdPath())
  local cmd = raw and decodeJson(raw) or nil
  if type(cmd) ~= 'table' then cmd = nil end
  local seq = cmd and tonumber(cmd.seq) or nil
  local hb = cmd and tonumber(cmd.heartbeat_mtime) or nil
  if seq and seq ~= lastCmdSeq then
    lastCmdSeq = seq
    cmdStaleAcc = 0
    cmdStaleLogged = false
  else
    cmdStaleAcc = cmdStaleAcc + step
  end
  -- A file left over from an old session has heartbeat_mtime far behind the wall clock.
  local ancient = hb ~= nil and (os.time() - hb) > CMD_DEAD_S
  if cmdStaleAcc > CMD_DEAD_S or ancient then
    if applying then
      -- Dead-man: the stream died while we held the car. Release it and disengage; Alt+A re-arms.
      releaseInputs('command stream dead')
      engaged = false
      fadeAcc = 0
      writeEngageFile('command_stream_dead')
      log('W', 'GVD', '[GVD] DISENGAGED: supervisor command stream dead')
      print('[GVD] DISENGAGED: supervisor command stream dead')
      pushUi()
    end
    return
  end
  if not cmd or cmd.engaged ~= true then
    -- Supervisor is not driving (has not read Alt+A yet / vetoed / shutting down): hands off.
    releaseInputs('supervisor not driving')
    return
  end
  if not arcadeQueued then arcadeQueued = queueVehicle(veh, VE_ARCADE) end
  -- Judged before we push the next command, and only while we actually hold the car: wheel
  -- chatter must not disengage, a real pedal or a sustained wheel pull must. Sticky like the
  -- supervisor's own check — Alt+A re-arms.
  if applying then
    local ch = ovrCheck(step)
    if ch ~= 'none' then
      releaseInputs('driver override (' .. ch .. ')')
      engaged = false
      fadeAcc = 0
      writeEngageFile('driver_override')
      log('W', 'GVD', '[GVD] DISENGAGED: driver override on ' .. ch)
      print('[GVD] DISENGAGED: driver override on ' .. ch)
      pushUi()
      return
    end
  end
  if cmdStaleAcc > CMD_STALE_S then
    -- Stream hiccup: straighten + brake, no throttle, until fresh commands resume.
    applying = true
    applyInputs(veh, 0, 0, 1)
    if not cmdStaleLogged then
      cmdStaleLogged = true
      log('W', 'GVD', '[GVD] cmd stale > ' .. tostring(CMD_STALE_S) .. 's: brake hold')
    end
    return
  end
  applying = true
  if applyInputs(veh, cmd.steer, cmd.throttle, cmd.brake) then
    lastAppliedSeq = seq or lastAppliedSeq
  end
end

-- M6 retail: the Python supervisor writes gvd_engage.json {"engaged":false} when it vetoes,
-- loses its heartbeat, or exits (finally block). Adopt that OFF so the HUD/ribbon never stay ON
-- without a supervisor. Never adopt ON from the file — engage always starts in-game (Alt+A / GVD app).
local function syncEngageFromSupervisor()
  if not engaged then return end
  local raw = readText(userEngagePath())
  if not raw then return end
  local f = decodeJson(raw)
  if type(f) ~= 'table' or f.engaged ~= false then return end
  -- Our own toggle write carries mtime = os.time(); only a supervisor write at/after it may switch us off.
  local mt = tonumber(f.mtime) or 0
  if mt < lastEngageWriteUnix then return end
  engaged = false
  fadeAcc = 0
  releaseInputs('supervisor disengaged')
  -- Reason: the engage file carries it since M6 (veto / driver_override / shutdown); fall back to state.
  local why = 'supervisor off'
  local fr = f.disengage_reason and tostring(f.disengage_reason) or nil
  if fr and fr ~= 'none' and fr ~= 'not_engaged' then
    why = fr
  elseif lastGood and lastGood.disengage_reason then
    local r = tostring(lastGood.disengage_reason)
    if r ~= 'none' and r ~= 'not_engaged' then why = r end
  end
  -- Later supervisor ticks only write the generic not_engaged, so keep the real reason for the HUD.
  luaDisengageReason = why
  log('I', 'GVD', '[GVD] DISENGAGED by supervisor (' .. why .. ')')
  print('[GVD] DISENGAGED by supervisor (' .. why .. ')')
  pushUi()
end

local function pollStateFile()
  local raw = readText(userStatePath())
  if not raw then return end
  local st = decodeJson(raw)
  if not st then return end
  lastGood = st
  readOverrideCfg(st)  -- config/control.yaml override: block, mirrored by the supervisor
  noteHeartbeat(st)
  local beat = tonumber(st.heartbeat_mtime)
  if beat ~= lastStateBeat then
    lastStateBeat = beat
    stateBeatAcc = 0
  end
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

local function pollState(dt)
  pollAcc = pollAcc + (dt or 0)
  stateBeatAcc = stateBeatAcc + (dt or 0)
  if pollAcc < pollEvery then return end
  pollAcc = 0
  pollStateFile()
  -- After the state read so the logged reason is the supervisor's fresh disengage_reason.
  syncEngageFromSupervisor()
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
  pollEgo(dt)
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
  log('I', 'GVD', string.format(
    '[GVD] driver override: steer deadband %.2f (force-feedback chatter must not disengage), pedals hard at brake %.2f / throttle %.2f',
    OVR.steer_deadband, OVR.brake_enter, OVR.throttle_enter))
  print('[GVD] loaded. Alt+A engage. Writes ' .. userEngagePath()
    .. '; drives the player vehicle from gvd_cmd.json while engaged (retail). BeamNGpy direct control on Tech.')
end

function M.onExtensionUnloaded()
  releaseInputs('extension unloaded')
  engaged = false
  writeEngageFile('extension_unloaded')
  lastGood = nil
  log('I', 'GVD', '[GVD] unloaded.')
end

function M.toggleEngage()
  engaged = not engaged
  fadeAcc = 0
  if not engaged then releaseInputs('Alt+A disengage') end
  writeEngageFile()  -- a deliberate Alt+A is the normal path, not a reason the HUD should surface
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
