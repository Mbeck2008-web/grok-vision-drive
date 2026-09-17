-- Offline harness for gvdDocsDir dual-path sandbox (no BeamNG).
-- Extracts the live helpers from main.lua and asserts:
--   GVD_DOCS_DIR → override
--   FS running userfolder → matching product current\Documents\GVD
--     Tech  → …/BeamNG/BeamNG.tech/current/Documents/GVD
--     Drive → …/BeamNG/BeamNG.drive/current/Documents/GVD
--   env LOCALAPPDATA without FS → Drive default (retail); GVD_BEAMNG=1 → Tech
--   never Drive FS → Tech tail, never USERPROFILE/Documents/GVD,
--   never a bare gvd_*.json under current\
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
check(body:find('_localAppFromPath', 1, true), 'extracted _localAppFromPath')
check(body:find('_techCurrentFromPath', 1, true), 'extracted _techCurrentFromPath')
check(body:find('_driveCurrentFromPath', 1, true), 'extracted _driveCurrentFromPath')
check(body:find('GVD_DOCS_DIR', 1, true), 'GVD_DOCS_DIR override is in helpers')
check(body:find('LOCALAPPDATA', 1, true), 'LOCALAPPDATA product sandbox is in helpers')
check(body:find('BeamNG/BeamNG.tech/current/Documents/GVD', 1, true), 'Tech tail is in helpers')
check(body:find('BeamNG/BeamNG.drive/current/Documents/GVD', 1, true), 'Drive tail is in helpers')
check(body:find('_tryEnvDocs() or _tryFsDocs() or _tryEnvProductDocs()', 1, true),
  'gvdDocsDir: override, then FS product, then env LOCALAPPDATA')
check(not body:find('USERPROFILE') or body:find('AppData/Local', 1, true),
  'USERPROFILE only synthesizes AppData/Local, not Documents/GVD')

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
  local chunk = body .. '\nreturn { localApp = _localAppFromPath, techCurrent = _techCurrentFromPath, docsDir = gvdDocsDir, file = gvdFile }\n'
  local fn, err = loadfn(chunk)
  check(fn, 'helpers compile: ' .. tostring(err))
  if setfenv then setfenv(fn, sandbox) else
    -- Lua 5.2+: inject env via _ENV in a wrapper. 5.1/luajit is the target.
    error('FAIL: setfenv required (lua5.1 / luajit)')
  end
  return fn()
end

local H = loadHelpers({}, nil)
local TECH = 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current/Documents/GVD'
local DRIVE = 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current/Documents/GVD'

-- ── localApp / techCurrent ───────────────────────────────────────────────────
check(H.localApp('C:/Users/Name/AppData/Local/BeamNG.drive/0.36') == 'C:/Users/Name/AppData/Local',
  'AppData/Local prefix from Drive userfolder')
check(H.localApp('C:\\Users\\Name\\AppData\\Local\\BeamNG\\BeamNG.tech\\current') == 'C:/Users/Name/AppData/Local',
  'AppData/Local prefix with backslashes')
check(H.localApp('C:/Users/Name/Documents/BeamNG.drive/current') == nil,
  'Documents userfolder does not invent LOCALAPPDATA')
check(H.localApp(nil) == nil, 'nil path → nil')
check(H.localApp('') == nil, 'empty path → nil')
check(H.techCurrent('C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current')
    == 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current',
  'nested BeamNG.tech/current is the Tech userfolder')
check(H.techCurrent('C:/Users/Name/AppData/Local/BeamNG.tech/current') == nil,
  'legacy BeamNG.tech (no BeamNG parent) is not the spec current')
check(H.techCurrent('C:/Users/Name/OneDrive/AppData/Local/BeamNG/BeamNG.tech/current') == nil,
  'OneDrive tech current is rejected')
check(H.localApp('C:/Users/Name/OneDrive/AppData/Local/foo') == nil,
  'OneDrive AppData/Local is rejected')

-- ── resolve ──────────────────────────────────────────────────────────────────
local function resolve(envVars, fs)
  return loadHelpers(envVars, fs).docsDir()
end

check(resolve({ GVD_DOCS_DIR = 'D:\\custom\\GVD', LOCALAPPDATA = 'C:\\Users\\Name\\AppData\\Local' }, nil)
    == 'D:/custom/GVD',
  'GVD_DOCS_DIR wins over LOCALAPPDATA')

check(resolve({ LOCALAPPDATA = 'C:\\Users\\Name\\AppData\\Local', USERPROFILE = 'C:\\Users\\Name' }, {
      getUserPath = function() return 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current' end,
      directoryCreate = function() end,
    }) == DRIVE,
  'FS Drive current wins over LOCALAPPDATA (not Tech tail)')

check(resolve({ LOCALAPPDATA = 'C:\\Users\\Name\\AppData\\Local', USERPROFILE = 'C:\\Users\\Name', GVD_BEAMNG = '1' }, {
      getUserPath = function() return 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current' end,
      directoryCreate = function() end,
    }) == DRIVE,
  'running Drive userfolder wins over GVD_BEAMNG env (Steam has no env)')

check(resolve({ USERPROFILE = 'C:\\Users\\Name' }, nil) == DRIVE,
  'USERPROFILE synthesizes AppData/Local Drive sandbox by default (not Documents/GVD)')

check(resolve({ USERPROFILE = 'C:\\Users\\Name', GVD_BEAMNG = '1' }, nil) == TECH,
  'USERPROFILE + GVD_BEAMNG=1 synthesizes Tech sandbox')

check(resolve({ HOME = '/home/me' }, nil)
    == '/home/me/AppData/Local/BeamNG/BeamNG.drive/current/Documents/GVD',
  'HOME synthesizes AppData/Local Drive sandbox when LOCALAPPDATA empty')

check(resolve({ LOCALAPPDATA = 'C:/Users/Name/OneDrive', USERPROFILE = 'C:/Users/Name' }, nil) == DRIVE,
  'OneDrive LOCALAPPDATA is rejected; USERPROFILE AppData/Local Drive sandbox')

check(resolve({}, {
      getUserPath = function() return 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current' end,
      directoryCreate = function() end,
    }) == TECH,
  'FS Tech current userfolder → Tech Documents/GVD')

check(resolve({ LOCALAPPDATA = 'C:/Users/Name/AppData/Local' }, {
      getUserPath = function() return 'C:/Users/Name/AppData/Local/BeamNG.drive/0.36' end,
      directoryCreate = function() end,
    }) == DRIVE,
  'FS Drive AppData userfolder reconstructs Drive current Documents/GVD (not Tech)')

check(resolve({}, {
      virtual2Native = function(_, vp)
        if vp == 'settings' then return 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current/settings' end
      end,
      directoryCreate = function() end,
    }) == TECH,
  'FS:virtual2Native Tech settings path → Tech Documents/GVD')

check(resolve({}, {
      virtual2Native = function(_, vp)
        if vp == 'settings' then return 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current/settings' end
      end,
      directoryCreate = function() end,
    }) == DRIVE,
  'FS:virtual2Native Drive settings path → Drive Documents/GVD')

check(resolve({ LOCALAPPDATA = 'C:/Users/Name/AppData/Local' }, {
      getUserPath = function() return 'C:/Users/Name/Documents/BeamNG.drive/0.36/current' end,
      directoryCreate = function() end,
    }) == DRIVE,
  'FS Documents Drive userfolder reconstructs Drive sandbox (not USERPROFILE/Documents/GVD, not Tech)')

check(resolve({}, {
      getUserPath = function() return 'C:/Users/Name/Documents/BeamNG.drive/0.36/current' end,
      directoryCreate = function() end,
    }) == 'Documents/GVD',
  'FS Documents userfolder without LOCALAPPDATA does NOT become USERPROFILE/Documents/GVD')

check(resolve({ LOCALAPPDATA = 'C:/Users/Name/AppData/Local' }, nil) == DRIVE,
  'LOCALAPPDATA without FS defaults to Drive/retail sandbox')

check(resolve({ LOCALAPPDATA = 'C:/Users/Name/AppData/Local', GVD_BEAMNG = '1' }, nil) == TECH,
  'LOCALAPPDATA + GVD_BEAMNG=1 (no FS) → Tech sandbox')

check(resolve({ LOCALAPPDATA = 'C:/Users/Name/AppData/Local', GVD_PRODUCT = 'tech' }, nil) == TECH,
  'LOCALAPPDATA + GVD_PRODUCT=tech → Tech sandbox')

check(resolve({ LOCALAPPDATA = 'C:/Users/Name/AppData/Local', GVD_BACKEND = 'beamngpy' }, nil) == TECH,
  'LOCALAPPDATA + GVD_BACKEND=beamngpy → Tech sandbox')

-- last resort is Documents/GVD, never a bare filename, never userfolder current\
local last = resolve({}, { directoryCreate = function() end })
check(last == 'Documents/GVD', 'last resort is Documents/GVD (not CWD, not current\\)')
check(last ~= 'current' and not last:match('gvd_.*%.json$'),
  'last resort is not a bare gvd_*.json')

-- gvdFile never returns a bare filename (Drive default without FS)
local files = loadHelpers({ LOCALAPPDATA = 'C:/Users/Name/AppData/Local' }, { directoryCreate = function() end })
for _, name in ipairs({'gvd_state.json', 'gvd_ego.json', 'gvd_engage.json', 'gvd_cmd.json'}) do
  local p = files.file(name)
  check(p == DRIVE .. '/' .. name, 'gvdFile(' .. name .. ') under Drive current/Documents/GVD by default')
  check(not p:match('^gvd_'), name .. ' is not a bare filename')
  check(not p:match('/current/' .. name .. '$'), name .. ' is not under userfolder current\\')
end

local techFiles = loadHelpers({ LOCALAPPDATA = 'C:/Users/Name/AppData/Local', GVD_BEAMNG = '1' }, { directoryCreate = function() end })
check(techFiles.file('gvd_state.json') == TECH .. '/gvd_state.json',
  'gvdFile under Tech current/Documents/GVD when GVD_BEAMNG=1')

-- mkdir is attempted on the resolved docs dir
local mkdirDir
local made = loadHelpers({ LOCALAPPDATA = 'C:/Users/Name/AppData/Local' }, {
  directoryCreate = function(_, dir) mkdirDir = dir end,
})
made.docsDir()
check(mkdirDir == DRIVE, 'mkdir of resolved Drive Documents/GVD by default')

check(body:find("docs dir=", 1, true) and body:find('gvdDocsLogged', 1, true),
  'one-shot log of resolved docs dir')

print('test_gvd_docs_dir: OK')
