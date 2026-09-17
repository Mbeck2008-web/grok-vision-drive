-- Offline harness for gvdDocsDir / _stripToUserHome (no BeamNG).
-- Extracts the live helpers from main.lua and asserts:
--   USERPROFILE prefer → …/Documents/GVD
--   FS paths under /Documents/ strip to Users/Name, then Documents/GVD
--   never a bare gvd_*.json under BeamNG userfolder current\
-- Run from repo root: lua5.1 scripts/test_gvd_docs_dir.lua

local function check(cond, msg)
  if not cond then error('FAIL: ' .. msg, 2) end
  print('ok   ' .. msg)
end

local f = assert(io.open('beamng_mod/lua/ge/extensions/gvd/main.lua', 'r'))
local src = f:read('*a')
f:close()

local i = src:find('local gvdDocsResolved', 1, true)
local j = src:find('\nlocal function userEngagePath', i, true)
check(i and j, 'main.lua exports gvdDocsDir helpers before userEngagePath')
local body = src:sub(i, j - 1)
check(body:find('_stripToUserHome', 1, true), 'extracted _stripToUserHome')
check(body:find('/Documents/', 1, true), 'Documents-parent strip is in helpers')
check(body:find('USERPROFILE', 1, true), 'USERPROFILE prefer is in helpers')

local loadfn = loadstring or load
local function loadHelpers(envVars, fs)
  local sandbox = {
    os = {
      getenv = function(k)
        if envVars == nil then return nil end
        return envVars[k]
      end,
    },
    FS = fs,
    log = function() end,
    print = function() end,
    pcall = pcall,
    tostring = tostring,
    ipairs = ipairs,
    pairs = pairs,
    type = type,
    string = string,
    table = table,
    math = math,
    assert = assert,
  }
  local chunk = body .. '\nreturn { strip = _stripToUserHome, docsDir = gvdDocsDir, file = gvdFile }\n'
  local fn, err = loadfn(chunk)
  check(fn, 'helpers compile: ' .. tostring(err))
  if setfenv then setfenv(fn, sandbox) else
    -- Lua 5.2+: inject env via _ENV in a wrapper. 5.1/luajit is the target.
    error('FAIL: setfenv required (lua5.1 / luajit)')
  end
  return fn()
end

-- Python mirror of Lua patterns (kept in lockstep by executing the extracted function).
local H = loadHelpers({}, nil)

-- ── strip: AppData (existing) ────────────────────────────────────────────────
check(H.strip('C:/Users/Name/AppData/Local/BeamNG.drive/0.36') == 'C:/Users/Name',
  'AppData/Local strips to Users/Name')
check(H.strip('C:\\Users\\Name\\AppData\\Roaming\\BeamNG.drive') == 'C:/Users/Name',
  'AppData/Roaming backslashes strip to Users/Name')
check(H.strip('C:/Users/Name/AppData/LocalLow/x') == 'C:/Users/Name',
  'AppData (generic) strips to Users/Name')

-- ── strip: Documents parent (the hotfix) ─────────────────────────────────────
check(H.strip('C:/Users/Name/Documents/BeamNG.drive/0.36/current') == 'C:/Users/Name',
  'Documents/BeamNG userfolder strips to Users/Name')
check(H.strip('C:\\Users\\Name\\Documents\\BeamNG.drive\\0.36\\current\\') == 'C:/Users/Name',
  'Documents userfolder with backslashes strips to Users/Name')
check(H.strip('C:/Users/Name/Documents/GVD/gvd_state.json') == 'C:/Users/Name',
  'path already under Documents/GVD strips to Users/Name')
check(H.strip('C:/Users/Name/Documents') == 'C:/Users/Name',
  'path ending at Documents strips to Users/Name')
check(H.strip('C:/Users/Name/Documents/') == 'C:/Users/Name',
  'Documents/ with trailing slash strips to Users/Name')
check(H.strip('C:/Users/Name/OneDrive/Documents/BeamNG.drive/current') == 'C:/Users/Name/OneDrive',
  'OneDrive Documents parent is OneDrive (not Users/Name/Documents/BeamNG...)')
check(H.strip('/home/me/Documents/BeamNG.drive/current') == '/home/me',
  'POSIX Documents parent strips to $HOME')
check(H.strip('C:/Users/Name/Documents/BeamNG.drive/current/Documents/GVD') == 'C:/Users/Name',
  'nested Documents under current still strips to the first parent (Users/Name)')
check(H.strip(nil) == nil, 'nil path → nil')
check(H.strip('') == nil, 'empty path → nil')
check(H.strip('D:/BeamNG.drive/current') == nil,
  'userfolder with no Documents/AppData does not invent a home')

-- ── resolve: USERPROFILE prefer → …/Documents/GVD ────────────────────────────
local function resolve(envVars, fs)
  return loadHelpers(envVars, fs).docsDir()
end

check(resolve({ USERPROFILE = 'C:\\Users\\Name', HOME = '/home/other' }, {
      getUserPath = function() return 'C:/Users/Name/Documents/BeamNG.drive/0.36/current' end,
      directoryCreate = function() end,
    }) == 'C:/Users/Name/Documents/GVD',
  'USERPROFILE wins over HOME and Documents FS path → …/Documents/GVD')

check(resolve({ HOME = '/home/me' }, {
      getUserPath = function() return '/home/me/Documents/BeamNG.drive/current' end,
      directoryCreate = function() end,
    }) == '/home/me/Documents/GVD',
  'HOME when USERPROFILE empty → …/Documents/GVD')

check(resolve({ LOCALAPPDATA = 'C:\\Users\\Name\\AppData\\Local' }, nil) == 'C:/Users/Name/Documents/GVD',
  'LOCALAPPDATA strip when USERPROFILE/HOME empty → …/Documents/GVD')

check(resolve({}, {
      getUserPath = function() return 'C:/Users/Name/Documents/BeamNG.drive/0.36/current' end,
      directoryCreate = function() end,
    }) == 'C:/Users/Name/Documents/GVD',
  'FS Documents userfolder (empty env, live Tech NO LINK bug) → Users/Name/Documents/GVD')

check(resolve({}, {
      getUserPath = function() return 'C:/Users/Name/AppData/Local/BeamNG.drive/0.36' end,
      directoryCreate = function() end,
    }) == 'C:/Users/Name/Documents/GVD',
  'FS AppData userfolder → Users/Name/Documents/GVD')

check(resolve({}, {
      virtual2Native = function(_, vp)
        if vp == 'settings' then return 'C:/Users/Name/Documents/BeamNG.drive/0.36/settings' end
      end,
      directoryCreate = function() end,
    }) == 'C:/Users/Name/Documents/GVD',
  'FS:virtual2Native Documents path → Users/Name/Documents/GVD')

-- last resort is Documents/GVD, never a bare filename, never userfolder current\
local last = resolve({}, { directoryCreate = function() end })
check(last == 'Documents/GVD', 'last resort is Documents/GVD (not CWD, not current\\)')
check(last ~= 'current' and not last:match('gvd_.*%.json$'),
  'last resort is not a bare gvd_*.json')

-- gvdFile never returns a bare filename
local files = loadHelpers({ USERPROFILE = 'C:/Users/Name' }, { directoryCreate = function() end })
for _, name in ipairs({'gvd_state.json', 'gvd_ego.json', 'gvd_engage.json', 'gvd_cmd.json'}) do
  local p = files.file(name)
  check(p == 'C:/Users/Name/Documents/GVD/' .. name, 'gvdFile(' .. name .. ') under Documents/GVD')
  check(not p:match('^gvd_'), name .. ' is not a bare filename')
  check(not p:match('/current/' .. name .. '$'), name .. ' is not under userfolder current\\')
end

-- mkdir is attempted on the resolved docs dir
local mkdirDir
local made = loadHelpers({ USERPROFILE = 'C:/Users/Name' }, {
  directoryCreate = function(_, dir) mkdirDir = dir end,
})
made.docsDir()
check(mkdirDir == 'C:/Users/Name/Documents/GVD', 'mkdir of resolved Documents/GVD')

-- one-shot log of resolved docs dir (source contract; log is called from gvdDocsDir)
check(body:find("docs dir=", 1, true) and body:find('gvdDocsLogged', 1, true),
  'one-shot log of resolved docs dir')

print('test_gvd_docs_dir: OK')
