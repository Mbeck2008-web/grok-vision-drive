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
local j = src:find('\nlocal function writeText', i, true)
check(i and j, 'main.lua exports gvdDocsDir helpers before writeText')
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
  local chunk = body .. '\nreturn { abs = _isAbsDiskPath, docsDir = gvdDocsDir, file = gvdFile, bus = luaBusPath, product = gvdProduct, same = busesSame }\n'
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
check(H.same(
      'C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current/Documents/GVD',
      'C:\\Users\\Name\\AppData\\Local\\BeamNG\\BeamNG.drive\\current\\Documents\\GVD') == true,
  'slash/case normalized same folder')
check(H.same(
      'C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current/Documents/GVD',
      'C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current/Documents/GVD') == false,
  'Drive vs Tech is not the same folder')
check(H.same(
      'C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current/Documents/GVD',
      'Documents/GVD') == false,
  'do not guess that relative Documents/GVD equals an absolute sandbox')
check(H.same(
      'C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current/Documents/GVD',
      'C:/Users/Name/OneDrive/Documents/GVD') == false,
  'OneDrive is not the GVD bus')

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

check(loadHelpers({}, {
      getUserPath = function() return 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current/' end,
      directoryCreate = function() end,
    }).bus() == 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current/Documents/GVD',
  'lua_bus resolves Documents/GVD via the running userfolder')

check(loadHelpers({}, {
      getUserPath = function() return 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current' end,
      directoryCreate = function() end,
    }).product() == 'tech',
  'product=tech from userfolder')

check(loadHelpers({}, {
      getUserPath = function() return 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.drive/current' end,
      directoryCreate = function() end,
    }).product() == 'drive',
  'product=drive from userfolder')

check(loadHelpers({}, { directoryCreate = function() end }).bus() == 'Documents/GVD',
  'lua_bus stays relative Documents/GVD when userfolder is unknown (do not guess)')

check(resolve({ LOCALAPPDATA = 'C:/Users/Name/AppData/Local', GVD_BEAMNG = '1' }, nil)
    == 'Documents/GVD',
  'LOCALAPPDATA + GVD_BEAMNG does not join the Tech tail for Lua')

local last = resolve({}, { directoryCreate = function() end })
check(last == 'Documents/GVD', 'default is Documents/GVD (not CWD, not current\\)')
check(last ~= 'current' and not last:match('gvd_.*%.json$'),
  'default is not a bare gvd_*.json')

local files = loadHelpers({ LOCALAPPDATA = 'C:/Users/Name/AppData/Local' }, { directoryCreate = function() end })
for _, name in ipairs({'gvd_state.json', 'gvd_ego.json', 'gvd_engage.json', 'gvd_cmd.json', 'gvd_link.json'}) do
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
check(body:find('luaBusPath', 1, true) and body:find('busesSame', 1, true),
  'identity helpers live next to gvdDocsDir')
check(not body:find('io.open', 1, true), 'identity/docs helpers have no io.open')

-- Tech 0.39.4 lua_bus write: global writeFile, jsonWriteFile, relative io.open.
-- The FS write method is nil on that build and must not be the writer.
local wi = src:find('\nlocal function writeText', 1, true)
local we = src:find('\nlocal lastEngageWriteUnix', wi, true)
check(wi and we, 'writeText is exported before engage writes')
local writeSrc = src:sub(wi + 1, we - 1)
local writeExec = writeSrc:gsub('%-%-[^\n]*', '')
check(not writeExec:find('FS:writeFile', 1, true), 'lua_bus write path has no FS:writeFile')
check(not writeExec:find('ret ~= false', 1, true), 'nil is not counted as success')
check(writeExec:find('if ok and ret then return true end', 1, true), 'writeFile/jsonWriteFile need a truthy return')
check(writeExec:find('if okW and wrote then return true end', 1, true), 'fh:write needs a truthy return')
check(writeExec:find('writeFile', 1, true), 'global writeFile is the Tech writer')
check(writeExec:find('jsonWriteFile', 1, true), 'jsonWriteFile is the second writer')
check(writeExec:find('io.open', 1, true), 'relative io.open is the last writer')
check(writeExec:find('_isAbsDiskPath', 1, true), 'absolute paths are refused')
check(not src:find('FS:writeFile', 1, true), 'main.lua has no FS:writeFile')

local TECH_BUS = 'C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current/Documents/GVD'
local linkPayload = '{"lua_bus":"' .. TECH_BUS .. '"}'
local writeChunk = body .. '\n' .. writeSrc .. '\nreturn writeText\n'
local function loadWriter(extra)
  local sandbox = {
    pcall = pcall,
    tostring = tostring,
    type = type,
    string = string,
    pairs = pairs,
    ipairs = ipairs,
    table = table,
    math = math,
    assert = assert,
    FS = {
      directoryCreate = function() end,
      writeFile = function()
        error('FS write method must not run')
      end,
    },
    io = { open = function() return nil end },
    writeFile = nil,
    jsonWriteFile = nil,
    jsonDecode = nil,
  }
  if extra then
    for k, v in pairs(extra) do sandbox[k] = v end
  end
  local fn, err = loadfn(writeChunk)
  check(fn, 'writeText compiles: ' .. tostring(err))
  if setfenv then setfenv(fn, sandbox) else
    error('FAIL: setfenv required (lua5.1 / luajit)')
  end
  return fn()
end

local vfsSeen = {}
local writer = loadWriter({
  writeFile = function(path, data)
    vfsSeen.path = path
    vfsSeen.data = data
    return true
  end,
  io = { open = function() error('io.open must not run when writeFile works') end },
  jsonWriteFile = function() error('jsonWriteFile must not run when writeFile works') end,
})
check(writer('Documents/GVD/gvd_link.json', linkPayload) == true, 'writeFile stores gvd_link.json')
check(vfsSeen.path == 'Documents/GVD/gvd_link.json', 'write lands under relative Documents/GVD')
check(vfsSeen.data == linkPayload and vfsSeen.data:find(TECH_BUS, 1, true),
  'lua_bus payload is the Tech Documents/GVD folder')
check(writer('C:/Users/Name/AppData/Local/BeamNG/BeamNG.tech/current/Documents/GVD/gvd_link.json', linkPayload) == false,
  'absolute Tech path is not a Lua write')
check(writer('gvd_link.json', linkPayload) == false, 'bare gvd_link.json is not a Lua write')

local jsonSeen = {}
local jsonWriter = loadWriter({
  writeFile = function() error('writeFile down') end,
  jsonDecode = function(s)
    local bus = tostring(s):match('"lua_bus"%s*:%s*"([^"]*)"')
    if not bus then return nil end
    return { lua_bus = bus }
  end,
  jsonWriteFile = function(path, obj)
    jsonSeen.path = path
    jsonSeen.bus = obj.lua_bus
    return true
  end,
  io = { open = function() error('io.open must not run when jsonWriteFile works') end },
})
check(jsonWriter('Documents/GVD/gvd_link.json', linkPayload) == true, 'jsonWriteFile fallback stores the link')
check(jsonSeen.path == 'Documents/GVD/gvd_link.json', 'jsonWriteFile path is Documents/GVD/gvd_link.json')
check(jsonSeen.bus == TECH_BUS, 'jsonWriteFile keeps the Tech lua_bus folder')

local ioSeen = {}
local ioWriter = loadWriter({
  io = {
    open = function(path, mode)
      ioSeen.path = path
      ioSeen.mode = mode
      return {
        write = function(_, data) ioSeen.data = data return true end,
        close = function() end,
      }
    end,
  },
})
check(ioWriter('Documents/GVD/gvd_link.json', linkPayload) == true, 'relative io.open fallback stores the link')
check(ioSeen.path == 'Documents/GVD/gvd_link.json' and ioSeen.mode == 'w',
  'io.open is the relative Documents/GVD path')
check(ioSeen.data == linkPayload, 'io.open writes the lua_bus payload')
check(ioWriter('C:/Abs/Documents/GVD/gvd_link.json', linkPayload) == false,
  'io.open fallback refuses an absolute path')

local function decodeLink(s)
  local bus = tostring(s):match('"lua_bus"%s*:%s*"([^"]*)"')
  if not bus then return nil end
  return { lua_bus = bus }
end

local nilSeen = {}
local nilWriter = loadWriter({
  writeFile = function() return nil end,
  jsonDecode = decodeLink,
  jsonWriteFile = function(path)
    nilSeen.path = path
    return true
  end,
  io = { open = function() error('io.open must not run after a nil writeFile fallthrough') end },
})
check(nilWriter('Documents/GVD/gvd_link.json', linkPayload) == true, 'nil writeFile falls through to jsonWriteFile')
check(nilSeen.path == 'Documents/GVD/gvd_link.json', 'nil writeFile used the next writer')

local falseSeen = {}
local falseWriter = loadWriter({
  writeFile = function() return false end,
  jsonDecode = decodeLink,
  jsonWriteFile = function() return false end,
  io = {
    open = function(path)
      falseSeen.path = path
      return {
        write = function(_, data) falseSeen.data = data return true end,
        close = function() end,
      }
    end,
  },
})
check(falseWriter('Documents/GVD/gvd_link.json', linkPayload) == true, 'false writeFile/jsonWriteFile fall through to io.open')
check(falseSeen.path == 'Documents/GVD/gvd_link.json', 'false returns reached relative io.open')
check(falseSeen.data == linkPayload, 'false fallthrough still writes the lua_bus payload')

local bytesWriter = loadWriter({
  writeFile = function() return 12 end,
  jsonWriteFile = function() error('jsonWriteFile must not run on a truthy byte count') end,
  io = { open = function() error('io.open must not run on a truthy byte count') end },
})
check(bytesWriter('Documents/GVD/gvd_link.json', linkPayload) == true, 'non-empty writeFile return is success')

local fhNil = loadWriter({
  io = {
    open = function()
      return {
        write = function() return nil end,
        close = function() end,
      }
    end,
  },
})
check(fhNil('Documents/GVD/gvd_link.json', linkPayload) == false, 'nil fh:write is not success')

local fhFalse = loadWriter({
  io = {
    open = function()
      return {
        write = function() return false end,
        close = function() end,
      }
    end,
  },
})
check(fhFalse('Documents/GVD/gvd_link.json', linkPayload) == false, 'false fh:write is not success')

local fhThrow = loadWriter({
  io = {
    open = function()
      return {
        write = function() error('disk full') end,
        close = function() end,
      }
    end,
  },
})
check(fhThrow('Documents/GVD/gvd_link.json', linkPayload) == false, 'throwing fh:write is not success')

print('test_gvd_docs_dir: OK')
