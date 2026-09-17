-- Offline harness for Lua relative Documents/GVD bus (no BeamNG).
-- Extracts the live helpers from main.lua and asserts:
--   gvdDocsDir → Documents/GVD (VFS userfolder mapping)
--   relative GVD_DOCS_DIR still wins; absolute GVD_DOCS_DIR is ignored
--     (absolute override is Python's Tech/Drive write root)
--   gvdFile → Documents/GVD/<name>, never a bare gvd_*.json under current\
--   never USERPROFILE/Documents/GVD, never LOCALAPPDATA Tech/Drive tails
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
check(body:find('GVD_DOCS_DIR', 1, true), 'GVD_DOCS_DIR override is in helpers')
check(body:find("dir = 'Documents/GVD'", 1, true), 'default docs dir is relative Documents/GVD')
check(body:find('_isAbsDiskPath', 1, true), 'absolute override is detected')
check(not body:find('_tryFsDocs', 1, true), 'Lua does not reconstruct FS Tech/Drive tails')
check(not body:find('_tryEnvProductDocs', 1, true), 'Lua does not join LOCALAPPDATA Tech/Drive tails')
check(not body:find('USERPROFILE', 1, true) or body:find('Python', 1, true),
  'USERPROFILE is not a Lua bus root')

-- readText: VFS / FS:readFile only — no absolute io.open
local ri = src:find('local function _gvdBusRel', 1, true)
local rj = src:find('\nlocal function decodeJson', ri, true)
check(ri and rj, 'bus read helpers are exported before decodeJson')
local readBody = src:sub(ri, rj - 1)
local readExec = readBody:gsub('%-%-[^\n]*', '')
check(readExec:find('FS:readFile', 1, true), 'readText uses FS:readFile')
check(readExec:find('readFile', 1, true), 'readText uses VFS readFile')
check(readExec:find('Documents/GVD', 1, true), 'readText prefers relative Documents/GVD')
check(not readExec:find('io.open', 1, true), 'readText has no io.open')
check(not src:find('_ioOpenRead', 1, true), 'no _ioOpenRead helper')
check(not src:find('_ioReadAll', 1, true), 'no _ioReadAll helper')
check(src:find("local path = STATE_REL", 1, true), 'pollStateFile reads STATE_REL')
check(src:find("readText(STATE_REL)", 1, true), 'state reads use STATE_REL')
check(src:find("readText(CMD_REL)", 1, true), 'cmd reads use CMD_REL')
check(src:find("readText(ENGAGE_REL)", 1, true), 'engage reads use ENGAGE_REL')

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
  local chunk = body .. '\nreturn { abs = _isAbsDiskPath, docsDir = gvdDocsDir, file = gvdFile }\n'
  local fn, err = loadfn(chunk)
  check(fn, 'helpers compile: ' .. tostring(err))
  if setfenv then setfenv(fn, sandbox) else
    error('FAIL: setfenv required (lua5.1 / luajit)')
  end
  return fn()
end

local H = loadHelpers({}, nil)
check(H.abs('C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current/Documents/GVD') == true,
  'Windows drive path is absolute')
check(H.abs('\\\\server\\share\\GVD') == true, 'UNC path is absolute')
check(H.abs('/tmp/custom/GVD') == true, 'POSIX path is absolute')
check(H.abs('Documents/GVD') == false, 'Documents/GVD is relative')
check(H.abs('custom/GVD') == false, 'custom relative dir is not absolute')

local function resolve(envVars, fs)
  return loadHelpers(envVars, fs).docsDir()
end

check(resolve({ GVD_DOCS_DIR = 'custom/GVD', LOCALAPPDATA = 'C:\\Users\\Name\\AppData\\Local' }, nil)
    == 'custom/GVD',
  'relative GVD_DOCS_DIR wins')

check(resolve({
      GVD_DOCS_DIR = 'C:\\Users\\Name\\AppData\\Local\\BeamNG\\BeamNG.tech\\current\\Documents\\GVD',
      LOCALAPPDATA = 'C:\\Users\\Name\\AppData\\Local',
      GVD_BEAMNG = '1',
    }, nil) == 'Documents/GVD',
  'absolute GVD_DOCS_DIR (Tech write root) is ignored for Lua')

check(resolve({
      LOCALAPPDATA = 'C:\\Users\\Name\\AppData\\Local',
      USERPROFILE = 'C:\\Users\\Name',
      GVD_BEAMNG = '1',
    }, {
      getUserPath = function() return 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current' end,
      directoryCreate = function() end,
    }) == 'Documents/GVD',
  'FS Tech current does not become an absolute Lua read path')

check(resolve({
      LOCALAPPDATA = 'C:\\Users\\Name\\AppData\\Local',
      USERPROFILE = 'C:\\Users\\Name',
    }, {
      getUserPath = function() return 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current' end,
      directoryCreate = function() end,
    }) == 'Documents/GVD',
  'FS Drive current does not become an absolute Lua read path')

check(resolve({ USERPROFILE = 'C:\\Users\\Name' }, nil) == 'Documents/GVD',
  'USERPROFILE does not become USERPROFILE/Documents/GVD')

check(resolve({ LOCALAPPDATA = 'C:/Users/Name/AppData/Local', GVD_BEAMNG = '1' }, nil)
    == 'Documents/GVD',
  'LOCALAPPDATA + GVD_BEAMNG does not join the Tech tail for Lua')

local last = resolve({}, { directoryCreate = function() end })
check(last == 'Documents/GVD', 'default is Documents/GVD (not CWD, not current\\)')
check(last ~= 'current' and not last:match('gvd_.*%.json$'),
  'default is not a bare gvd_*.json')

local files = loadHelpers({ LOCALAPPDATA = 'C:/Users/Name/AppData/Local' }, { directoryCreate = function() end })
for _, name in ipairs({'gvd_state.json', 'gvd_ego.json', 'gvd_engage.json', 'gvd_cmd.json'}) do
  local p = files.file(name)
  check(p == 'Documents/GVD/' .. name, 'gvdFile(' .. name .. ') is relative Documents/GVD')
  check(not p:match('^gvd_'), name .. ' is not a bare filename')
  check(not p:match('/current/' .. name .. '$'), name .. ' is not under userfolder current\\')
  check(not p:match('^%a:'), name .. ' is not an absolute disk path')
end

local mkdirDir
local made = loadHelpers({}, {
  directoryCreate = function(_, dir) mkdirDir = dir end,
})
made.docsDir()
check(mkdirDir == 'Documents/GVD', 'mkdir of relative Documents/GVD')

check(body:find("docs dir=", 1, true) and body:find('gvdDocsLogged', 1, true),
  'one-shot log of resolved docs dir')

print('test_gvd_docs_dir: OK')
