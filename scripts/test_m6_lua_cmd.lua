-- Offline harness for beamng_mod/lua/ge/extensions/gvd/main.lua retail drive (no BeamNG).
-- Run from repo root: lua5.1 scripts/test_m6_lua_cmd.lua   (luajit works too; needs setfenv/loadstring)
-- Stubs the GE globals and a player vehicle whose queueLuaCommand *executes* the vehicle-Lua snippet
-- in a sandbox with a recording input.event, electrics.values and obj:queueGameEngineLua round-trip.

local logs = {}
function log(level, tag, msg) logs[#logs + 1] = msg end

-- Minimal JSON decoder: flat scalars plus one level of nested objects, which is all the mod
-- reads (gvd_state.json nests override_cfg). BeamNG provides a real jsonDecode in-game.
local function decodeScalars(s)
  local t = {}
  for k, v in s:gmatch('"([%w_]+)"%s*:%s*([^,}]+)') do
    v = v:gsub('^%s+', ''):gsub('%s+$', '')
    if v == 'true' then t[k] = true
    elseif v == 'false' then t[k] = false
    elseif v == 'null' then t[k] = nil
    elseif v:sub(1, 1) == '"' then t[k] = v:sub(2, -2)
    else t[k] = tonumber(v) end
  end
  return t
end

function jsonDecode(s)
  local t = {}
  local rest = s:gsub('"([%w_]+)"%s*:%s*(%b{})', function(k, obj)
    t[k] = decodeScalars(obj)
    return ''
  end)
  for k, v in pairs(decodeScalars(rest)) do t[k] = v end
  return t
end

local home = os.getenv('HOME') or os.getenv('USERPROFILE')
assert(home and home ~= '', 'HOME/USERPROFILE required')
os.execute('mkdir -p "' .. home .. '/Documents/GVD"')
local docs = home .. '/Documents/GVD'
local engagePath = docs .. '/gvd_engage.json'
local statePath = docs .. '/gvd_state.json'
local cmdPath = docs .. '/gvd_cmd.json'
local egoPath = docs .. '/gvd_ego.json'

local function writeFile(p, s) local f = assert(io.open(p, 'w')); f:write(s); f:close() end
local function readFileAll(p) local f = io.open(p, 'r'); if not f then return nil end; local d = f:read('*a'); f:close(); return d end
os.remove(egoPath)

-- ── fake vehicle: executes queued vehicle-Lua in a sandbox ─────────────────────────────────────
local events = {}          -- recorded input.event calls, in order
local shifter = {}         -- drivetrain.setShifterMode calls
local geQueue = {}         -- strings vehicle Lua queued back to GE
local electricsValues = { wheelspeed = 0, steering_input = 0, throttle_input = 0, brake_input = 0 }

local veEnv = {
  input = { event = function(itype, ivalue, filter) events[#events + 1] = { itype, ivalue, filter } end },
  electrics = { values = electricsValues },
  drivetrain = { setShifterMode = function(mode) shifter[#shifter + 1] = mode end },
  obj = { queueGameEngineLua = function(_, s) geQueue[#geQueue + 1] = s end },
  pcall = pcall, tonumber = tonumber, tostring = tostring, string = string, math = math,
}
local Veh = {}
Veh.__index = Veh
function Veh:queueLuaCommand(code)
  local fn, err = loadstring(code, 'vehicle-lua')
  assert(fn, 'vehicle snippet does not compile: ' .. tostring(err) .. '\n' .. code)
  setfenv(fn, veEnv)
  self.env = self.env or veEnv
  -- tag recorded events with the vehicle id so a vehicle switch can be verified
  local n0 = #events
  fn()
  for i = n0 + 1, #events do events[i].veh = self.id end
end
function Veh:getPosition() return { x = 0, y = 0, z = 0 } end
function Veh:getID() return self.id end
local fakeVeh = setmetatable({ id = 101 }, Veh)
local otherVeh = setmetatable({ id = 202 }, Veh)
local currentVeh = fakeVeh
be = { getPlayerVehicle = function(_, _idx) return currentVeh end }

local M = dofile('beamng_mod/lua/ge/extensions/gvd/main.lua')
extensions = { gvd_main = M }

local function drainGE()
  -- run whatever vehicle Lua queued back into GE (extensions.gvd_main.onEgoFeedback(...))
  local q = geQueue
  geQueue = {}
  for _, s in ipairs(q) do
    local fn, err = loadstring(s, 'ge-lua')
    assert(fn, 'GE snippet does not compile: ' .. tostring(err) .. '\n' .. s)
    fn()
  end
end

local function lastEvent(itype)
  for i = #events, 1, -1 do if events[i][1] == itype then return events[i] end end
  return nil
end
local function clearEvents() events = {} end
local function check(cond, msg) if not cond then error('FAIL: ' .. msg, 2) end; print('ok   ' .. msg) end
local function near(a, b) return math.abs((a or 0) - (b or 0)) < 1e-6 end

local seq = 0
local function writeCmd(engaged, steer, throttle, brake, hbOffset)
  seq = seq + 1
  writeFile(cmdPath, string.format(
    '{"steer": %.4f, "throttle": %.4f, "brake": %.4f, "seq": %d, "engaged": %s, "heartbeat_mtime": %.3f, "reason": "ok"}',
    steer, throttle, brake, seq, engaged and 'true' or 'false', os.time() + (hbOffset or 0)))
  return seq
end
local function beatState(engaged, reason, overrideCfg)
  writeFile(statePath, string.format('{"engaged":%s,"disengage_reason":"%s","heartbeat_mtime":%.3f,"policy":"modular","loop_hz":15%s}',
    engaged and 'true' or 'false', reason or 'none', os.clock() + seq, overrideCfg and (',"override_cfg":' .. overrideCfg) or ''))
end

M.onExtensionLoaded()
beatState(false, 'not_engaged')
writeCmd(false, 0, 0, 1)
M.onUpdate(0.2)
check(#events == 0, 'not engaged in-game → vehicle untouched')

-- 1) Alt+A, but the supervisor has not read the flag yet → hands off (no brake tap on engage)
M.toggleEngage()
check(M.isEngaged(), 'Alt+A → engaged')
M.onUpdate(0.06)
check(#events == 0, 'supervisor cmd engaged=false → no input.event')

-- 2) supervisor engaged: real steer/throttle applied with stock filters, +steer passes through (right)
writeCmd(true, 0.3, 0.4, 0.0)
beatState(true)
M.onUpdate(0.06)
local st = lastEvent('steering'); local th = lastEvent('throttle'); local br = lastEvent('brake')
check(st and near(st[2], 0.3) and st[3] == 1, "input.event('steering', 0.3, 1) pad-smoothed, sign unchanged")
check(th and near(th[2], 0.4) and th[3] == 2, "input.event('throttle', 0.4, 2) direct")
check(br and near(br[2], 0.0) and br[3] == 2, "input.event('brake', 0, 2) direct")
check(#shifter == 1 and shifter[1] == 'arcade', 'arcade shifter mode queued once on first drive')

-- 3) electrics echo → GE → gvd_ego.json with applied seq
electricsValues.wheelspeed = 5.5
electricsValues.steering_input = 0.3
M.onUpdate(0.11)
drainGE()
local ego = readFileAll(egoPath)
check(ego and ego:find('"speed_mps":5.500') and ego:find('"applied_seq":' .. seq) and ego:find('"applying":true'),
  'gvd_ego.json carries wheelspeed + applied_seq + applying (' .. tostring(ego) .. ')')

-- 4) AEB: brake wins, throttle forced to 0 even if both were set
clearEvents()
writeCmd(true, 0.0, 0.5, 1.0)
M.onUpdate(0.06)
check(near(lastEvent('brake')[2], 1.0) and near(lastEvent('throttle')[2], 0.0), 'brake>0 → throttle forced 0')

-- 5) stale stream (>0.35 s without a new seq) → brake hold, steering straight
clearEvents()
writeCmd(true, 0.6, 0.5, 0.0)
M.onUpdate(0.06)
check(near(lastEvent('steering')[2], 0.6), 'fresh cmd applied (steer 0.6)')
clearEvents()
M.onUpdate(0.2); M.onUpdate(0.2)
local sh = lastEvent('steering'); local bh = lastEvent('brake'); local thh = lastEvent('throttle')
check(sh and near(sh[2], 0) and bh and near(bh[2], 1) and thh and near(thh[2], 0), 'stale > 0.35 s → steer 0 / throttle 0 / brake 1 hold')
check(M.isEngaged(), 'still engaged during short stall')

-- 6) stream resumes → normal apply again
clearEvents()
writeCmd(true, -0.2, 0.3, 0.0)
M.onUpdate(0.06)
check(near(lastEvent('steering')[2], -0.2) and near(lastEvent('throttle')[2], 0.3), 'fresh seq → resumes applying (left steer passes through)')

-- 7) Alt+A disengage → one release (all zeros), then hands off even if cmd says engaged
clearEvents()
M.toggleEngage()
check(not M.isEngaged(), 'Alt+A → disengaged')
check(near(lastEvent('steering')[2], 0) and near(lastEvent('throttle')[2], 0) and near(lastEvent('brake')[2], 0), 'release: steering/throttle/brake → 0')
clearEvents()
writeCmd(true, 0.5, 0.5, 0.0)
M.onUpdate(0.06)
check(#events == 0, 'disengaged in-game → stale engaged cmd ignored')

-- 8) supervisor-side disengage (veto/exit): cmd engaged=false + engage file false → release + OFF
M.toggleEngage()
writeCmd(true, 0.1, 0.2, 0.0)
M.onUpdate(0.06)
check(near(lastEvent('throttle')[2], 0.2), 'engaged again and driving')
clearEvents()
writeCmd(false, 0.0, 0.0, 1.0)
beatState(false, 'low_lane_conf')
writeFile(engagePath, string.format('{"engaged": false, "mtime": %.3f}', os.time() + 0.5))
M.onUpdate(0.11)
check(near(lastEvent('throttle')[2], 0) and near(lastEvent('brake')[2], 0), 'supervisor not driving → inputs released (no brake hold on the player)')
check(not M.isEngaged(), 'supervisor false adopted → OFF')

-- 8b) engage file reason wins over state (M6 write_engage_flag disengage_reason)
M.toggleEngage()
writeCmd(true, 0.1, 0.2, 0.0)
M.onUpdate(0.06)
writeCmd(false, 0.0, 0.0, 1.0)
beatState(false, 'none')
writeFile(engagePath, string.format('{"engaged": false, "mtime": %.3f, "disengage_reason": "driver_override"}', os.time() + 0.5))
M.onUpdate(0.11)
check(not M.isEngaged() and logs[#logs]:find('%(driver_override%)') ~= nil, 'engage-file disengage_reason shown (driver_override)')

-- 9) dead-man: supervisor dies while we hold the car → brake hold, then release + auto-disengage + file false
M.toggleEngage()
writeCmd(true, 0.2, 0.4, 0.0)
M.onUpdate(0.06)
check(M.isEngaged() and near(lastEvent('throttle')[2], 0.4), 'engaged, driving before the supervisor dies')
clearEvents()
for _ = 1, 3 do M.onUpdate(0.5) end  -- 1.5 s > CMD_DEAD_S (1.0)
check(not M.isEngaged(), 'stream dead > CMD_DEAD_S (1.0 s) → auto-disengaged')
check(near(lastEvent('brake')[2], 0) and near(lastEvent('throttle')[2], 0), 'dead-man ends with released inputs')
local ef = readFileAll(engagePath)
check(ef and ef:find('"engaged":false') ~= nil, 'dead-man wrote gvd_engage.json engaged=false')
local sawHold = false
for _, e in ipairs(events) do if e[1] == 'brake' and near(e[2], 1) then sawHold = true end end
check(sawHold, 'brake hold happened before the dead-man release')

-- 10) left-over file from an old session (heartbeat_mtime far in the past) is never applied
M.toggleEngage()
clearEvents()
writeCmd(true, 0.9, 0.9, 0.0, -100)
M.onUpdate(0.06)
check(#events == 0, 'ancient cmd (old session) ignored')
M.toggleEngage()

-- 11) player switches vehicle while GVD holds the wheel → old vehicle released, new one driven (+arcade)
M.toggleEngage()
writeCmd(true, 0.25, 0.35, 0.0)
M.onUpdate(0.06)
check(near(lastEvent('throttle')[2], 0.35) and lastEvent('throttle').veh == 101, 'driving vehicle 101')
clearEvents(); shifter = {}
currentVeh = otherVeh
writeCmd(true, 0.25, 0.35, 0.0)
M.onUpdate(0.06)
local releasedOld, droveNew = false, false
for _, e in ipairs(events) do
  if e.veh == 101 and e[1] == 'throttle' and near(e[2], 0) then releasedOld = true end
  if e.veh == 202 and e[1] == 'throttle' and near(e[2], 0.35) then droveNew = true end
end
check(releasedOld, 'previous vehicle 101 got release zeros')
check(droveNew, 'new vehicle 202 receives the drive command')
check(#shifter == 1 and shifter[1] == 'arcade', 'arcade re-armed for the new vehicle')
clearEvents()
M.toggleEngage()
check(lastEvent('throttle').veh == 202 and near(lastEvent('throttle')[2], 0), 'disengage releases the vehicle we actually drove')
currentVeh = fakeVeh

-- 13) player override: force-feedback chatter must not disengage, a real driver must ──────
-- The mod only sees the electrics echo, so the harness plays the wheel/pedals by writing
-- electrics.values; applyCmdJson reads the echo the *previous* tick round-tripped through GE.
local function echo(steer, thr, brk)
  electricsValues.steering_input = steer
  electricsValues.throttle_input = thr
  electricsValues.brake_input = brk
end
local function driveTick(steerCmd, thrCmd, brkCmd, overrideCfg)
  writeCmd(true, steerCmd, thrCmd, brkCmd)
  beatState(true, 'none', overrideCfg)
  M.onUpdate(0.11)
  drainGE()
end
local function warmDrive(steerCmd, thrCmd, brkCmd, overrideCfg)
  -- 3 * lpf_tau_ms = 240 ms of command before the residual is judged.
  for _ = 1, 4 do driveTick(steerCmd, thrCmd, brkCmd, overrideCfg) end
end

echo(0, 0.3, 0)
M.toggleEngage()
warmDrive(0, 0.3, 0)
check(M.isEngaged(), 'engaged and driving before the wheel is touched')

for i = 1, 16 do
  -- what a force-feedback wheel does over bumps: big, fast, alternating
  echo((i % 2 == 0) and 0.7 or -0.7, 0.3, 0)
  driveTick(0, 0.3, 0)
end
check(M.isEngaged(), 'alternating force-feedback chatter (+/-0.7) does not disengage')

for _ = 1, 12 do
  echo(0.02, 0.3, 0)
  driveTick(0, 0.3, 0)
end
check(M.isEngaged(), 'steady residual inside steer_exit does not disengage')

-- a one-tick kick then rest: spike reject, the filter never follows
echo(0.7, 0.3, 0)
driveTick(0, 0.3, 0)
echo(0, 0.3, 0)
for _ = 1, 6 do driveTick(0, 0.3, 0) end
check(M.isEngaged(), 'one-tick FFB kick does not disengage')

-- a held residual past enter, in one direction, for steer_hold_ms
echo(0.25, 0.3, 0)
for _ = 1, 8 do driveTick(0, 0.3, 0) end
check(not M.isEngaged(), 'steer residual held past enter in one direction -> player_steer')
check(logs[#logs]:find('player_steer') ~= nil, 'override logged player_steer')
local ef2 = readFileAll(engagePath)
check(ef2 and ef2:find('"engaged":false') and ef2:find('player_steer'),
  'override wrote gvd_engage.json engaged=false disengage_reason=player_steer')

-- pedals stay tight: no filter, no dwell
echo(0, 0.3, 0)
M.toggleEngage()
warmDrive(0, 0.3, 0)
check(M.isEngaged(), 'Alt+A re-arms after an override')
echo(0, 0.3, 0.12)
driveTick(0, 0.3, 0); driveTick(0, 0.3, 0)
check(not M.isEngaged(), 'player brake press (0.12) -> player_brake')
check(logs[#logs]:find('player_brake') ~= nil, 'override logged player_brake')

-- GVD's own AEB brake hold echoes back at 1.0 and must not read as the player
echo(0, 0, 1.0)
M.toggleEngage()
warmDrive(0, 0, 1.0)
driveTick(0, 0, 1.0); driveTick(0, 0, 1.0)
check(M.isEngaged(), 'our own brake=1 echoed back is not an override')

-- GVD's own steer echo is residual 0, even at 0.4 of lock
echo(0.4, 0.3, 0)
for _ = 1, 8 do
  echo(0.4, 0.3, 0)
  driveTick(0.4, 0.3, 0)
end
check(M.isEngaged(), 'wheel matching cmd.steer is not an override (residual, not absolute)')

-- thresholds are tunable: the supervisor mirrors config/control.yaml into gvd_state.override_cfg
local tuned = '{"steer_enter":0.02,"steer_exit":0.01,"steer_hold_ms":0,"steer_spike":1.0,"brake_enter":0.06,"throttle_enter":0.10,"lpf_tau_ms":0}'
echo(0, 0.3, 0)
driveTick(0, 0.3, 0, tuned)
check(M.isEngaged(), 'still engaged after adopting the mirrored thresholds')
check(logs[#logs]:find('enter 0.020') ~= nil or logs[#logs - 1]:find('enter 0.020') ~= nil,
  'mirrored override_cfg logged (enter 0.020)')
echo(0.05, 0.3, 0)
driveTick(0, 0.3, 0, tuned); driveTick(0, 0.3, 0, tuned)
check(not M.isEngaged(), 'tuned steer_enter 0.02 trips on a 0.05 residual the default 0.08 ignores')

-- swapping vehicles re-arms the warm-up: the new car's echo says nothing about the old car's commands
local pin = '{"steer_enter":0.08,"steer_exit":0.04,"steer_hold_ms":200,"steer_spike":0.20,"brake_enter":0.06,"throttle_enter":0.10,"lpf_tau_ms":80}'
echo(0, 0.3, 0)
M.toggleEngage()
warmDrive(0, 0.3, 0, pin)
check(M.isEngaged(), 'driving before the vehicle switch')
currentVeh = otherVeh
echo(0.25, 0.3, 0)
driveTick(0, 0.3, 0, pin); driveTick(0, 0.3, 0, pin)
check(M.isEngaged(), 'vehicle switch re-arms the override warm-up')
for _ = 1, 8 do driveTick(0, 0.3, 0, pin) end
check(not M.isEngaged(), 'once warm again a held residual is player_steer')
currentVeh = fakeVeh

-- 12) chrome check on everything we push to the vehicle / HUD
for _, e in ipairs(logs) do assert(not e:lower():find('tesla') and not e:find('FSD'), 'chrome in log: ' .. e) end

print('test_m6_lua_cmd: OK')
