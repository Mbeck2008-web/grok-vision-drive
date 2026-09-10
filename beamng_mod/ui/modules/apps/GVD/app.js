// GVD in-game app: engage + settings (path/ghosts, policy, GVD VISION screen).
// The rich VISION lexicon lives on the OpenCV second screen (python/viz/stage.py).
// Data comes from the existing gvdUi guihooks push (gvd/main.lua reads Documents/GVD/gvd_state.json).
// Titles stay GVD / VISION; scripts/test_gvd_ui_app.py fails the build on any other product chrome.
angular.module('beamng.apps')
.directive('gvdApp', [function () {
  return {
    templateUrl: '/ui/modules/apps/GVD/app.html',
    replace: true,
    restrict: 'EA',
    scope: true,
    link: function (scope, element) {
      var POLICIES = ['modular', 'e2e', 'shadow'];

      var ui = {
        engaged: false, applying: false, link: 'none', hbAge: null, disengageReason: null,
        showPath: true, showGhosts: false, showScene: false,
        policy: null, policyReq: null, e2eOk: null, vetoReason: null, e2eBackend: null,
        hz: null, camHz: null, ttc: null, aeb: null, n: null, speed: null,
        targetV: null, brakeCmd: null,
        pathConf: null, pathWidth: null, pathPreview: null, laneConf: null,
        camBackend: null, camNote: null, camOk: null, camTotal: null,
        vizWindow: null, vizScreen: null, vizNote: null, vizScreenReq: null,
        inferMs: null, rssMb: null, vramUsed: null, vramTotal: null, gpu: null,
        detector: null, actuator: null, cmdApplied: null, cmdReason: null,
        cmdSeq: null, cmdAckSeq: null, egoSource: null,
        clipTrigger: null, encodeBackend: null
      };
      // Cleared when the supervisor is gone: stale numbers must not look live.
      var TELEMETRY = [
        'hz', 'camHz', 'ttc', 'aeb', 'n', 'speed', 'pathConf', 'pathWidth', 'pathPreview',
        'laneConf', 'policy', 'e2eOk', 'vetoReason', 'e2eBackend', 'camBackend', 'camNote',
        'targetV', 'brakeCmd', 'camOk', 'camTotal', 'vizWindow', 'vizScreen', 'vizNote', 'inferMs', 'rssMb',
        'vramUsed', 'vramTotal', 'gpu', 'detector', 'actuator', 'cmdApplied', 'cmdReason',
        'cmdSeq', 'cmdAckSeq', 'egoSource', 'clipTrigger', 'encodeBackend', 'disengageReason'
      ];

      scope.ui = ui;
      scope.policies = POLICIES;
      scope.nerd = false;

      // ---------------------------------------------------------------- lua
      var luaWarned = false;
      function lua(cmd) {
        try {
          if (typeof bngApi !== 'undefined' && bngApi.engineLua) {
            bngApi.engineLua(cmd);
            return true;
          }
        } catch (e) {
          if (!luaWarned) { luaWarned = true; try { console.warn('[GVD] engineLua failed', e); } catch (e2) {} }
          return false;
        }
        if (!luaWarned) { luaWarned = true; try { console.warn('[GVD] bngApi.engineLua unavailable'); } catch (e2) {} }
        return false;
      }
      function gvdLua(call) {
        return lua('if extensions and extensions.gvd_main then extensions.gvd_main.' + call + ' end');
      }

      // ------------------------------------------------------------ actions
      scope.toggleEngage = function () { gvdLua('toggleEngage()'); };
      scope.setPath = function (v) { ui.showPath = !!v; gvdLua('setShowPath(' + (v ? 'true' : 'false') + ')'); };
      scope.setGhosts = function (v) { ui.showGhosts = !!v; gvdLua('setShowAgentGhosts(' + (v ? 'true' : 'false') + ')'); };
      scope.setPolicy = function (p) {
        if (POLICIES.indexOf(p) < 0) return;
        ui.policyReq = p;
        gvdLua("requestPolicy('" + p + "')");
      };
      scope.setVizScreen = function (s) {
        s = String(s);
        if (['auto', '1', '2', '3'].indexOf(s) < 0) return;
        ui.vizScreenReq = s;
        gvdLua("requestVizScreen('" + s + "')");
      };

      // ----------------------------------------------------------- readouts
      function dash(v) { return (v === null || v === undefined || v === '') ? '--' : v; }
      scope.dash = dash;

      scope.linkLabel = function () {
        if (ui.link === 'live') return 'link live';
        if (ui.link === 'stale') return 'link stale';
        return 'no link';
      };
      // Retail: the mod applies gvd_cmd.json itself (ui.applying). Tech: BeamNGpy drives the
      // vehicle directly, so the mod never applies and cmd_applied is the honest signal.
      function driving() {
        if (ui.link !== 'live' || !ui.engaged) return false;
        if (ui.applying) return true;
        return ui.actuator === 'beamngpy' && ui.cmdApplied === true;
      }
      scope.driving = driving;
      scope.rootClass = function () {
        return {
          'is-engaged': ui.engaged && ui.link === 'live',
          'is-drive': driving(),
          'is-hold': ui.engaged && ui.link !== 'live'
        };
      };
      scope.stateClass = function () {
        return {
          'is-on': ui.engaged && ui.link === 'live',
          'is-drive': driving(),
          'is-hold': ui.engaged && ui.link !== 'live'
        };
      };
      scope.stateLabel = function () {
        if (!ui.engaged) return 'DISENGAGED';
        if (ui.link !== 'live') return 'HOLD';
        return driving() ? 'DRIVE' : 'ENGAGED';
      };
      scope.stateReason = function () {
        if (ui.engaged && ui.link === 'none') return 'dead-man: no telemetry, actuators off';
        if (ui.engaged && ui.link === 'stale') return 'dead-man: heartbeat stale, actuators off';
        if (ui.engaged) {
          if (ui.aeb === 'brake') return 'AEB brake';
          if (ui.aeb === 'warn') return 'AEB warn';
          if (ui.vetoReason && ui.vetoReason !== 'none') return 'veto: ' + ui.vetoReason;
          if (driving()) return 'GVD holds the wheel - steer to take over';
          if (ui.actuator === 'cmd_json') return 'armed - waiting for supervisor commands';
          return 'armed - actuators idle';
        }
        var r = ui.disengageReason;
        if (r && r !== 'none' && r !== 'not_engaged') return 'last: ' + r;
        return 'standby';
      };

      scope.pathLine = function () {
        if (ui.link === 'none') return 'waiting for gvd_state.json';
        var low = ui.pathConf !== null && ui.pathConf < 0.45;
        var bits = ['conf ' + (ui.pathConf === null ? '--' : Number(ui.pathConf).toFixed(2)) + (low ? ' low' : '')];
        if (ui.pathWidth !== null && ui.pathWidth !== undefined) bits.push(Number(ui.pathWidth).toFixed(1) + ' m');
        if (ui.pathPreview) bits.push('preview path');
        if (ui.n !== null && ui.n !== undefined) bits.push(ui.n + ' tracked');
        return bits.join(' - ');
      };
      scope.segClass = function (p) {
        return { 'is-on': ui.policy === p, 'is-req': ui.policyReq === p && ui.policy !== p };
      };
      scope.policyLine = function () {
        if (ui.link === 'none') {
          return ui.policyReq ? ('requested ' + ui.policyReq + ' - supervisor not running') : 'supervisor not running';
        }
        var bits = ['e2e ' + dash(ui.e2eBackend)];
        bits.push('veto ' + dash(ui.vetoReason));
        if (ui.e2eOk === false) bits.push('e2e held by modular');
        if (ui.policyReq && ui.policyReq !== ui.policy) bits.push('requested ' + ui.policyReq + '...');
        return bits.join(' - ');
      };
      scope.camLine = function () {
        if (ui.link === 'none') return '--';
        var s = dash(ui.camBackend);
        if (ui.camTotal) s += ' - ' + ui.camOk + '/' + ui.camTotal + ' feeds';
        if (ui.camNote) s += ' - ' + ui.camNote;
        else if (ui.camBackend === 'window') s += ' - retail: cam_main only';
        return s;
      };
      scope.vizLine = function () {
        if (ui.link === 'none') return '--';
        if (!ui.vizWindow) return 'off - run python with --viz';
        return 'on - ' + dash(ui.vizNote || ('screen ' + dash(ui.vizScreen)));
      };
      scope.vizSel = function (s) {
        if (ui.vizScreenReq) return ui.vizScreenReq === s;
        return String(ui.vizScreen) === s;
      };
      scope.rateLine = function () {
        if (ui.link === 'none') return '--';
        return Math.round(ui.hz || 0) + ' Hz loop - ' + Math.round(ui.camHz || 0) + ' Hz cams - ' +
          (ui.inferMs === null ? '--' : Number(ui.inferMs).toFixed(1)) + ' ms infer';
      };
      scope.gpuLine = function () {
        if (ui.link === 'none') return '--';
        var vram = (ui.vramUsed === null ? '--' : Number(ui.vramUsed).toFixed(1)) + ' / ' +
          (ui.vramTotal === null ? '--' : Number(ui.vramTotal).toFixed(1)) + ' GB';
        return vram + (ui.gpu ? ' - ' + ui.gpu : '') + (ui.rssMb ? ' - rss ' + Math.round(ui.rssMb) + ' MB' : '');
      };
      scope.actuatorLine = function () {
        if (ui.link === 'none') return '--';
        var s = dash(ui.actuator);
        if (ui.applying) s += ' - mod driving';
        else if (ui.cmdApplied === false) s += ' - not applied';
        if (ui.cmdAckSeq != null && ui.cmdAckSeq >= 0) s += ' - ack ' + ui.cmdAckSeq + '/' + dash(ui.cmdSeq);
        if (ui.egoSource && ui.egoSource !== 'none') s += ' - ego ' + ui.egoSource;
        if (ui.cmdReason && ['ok', 'cmd_json_applied'].indexOf(ui.cmdReason) < 0) s += ' - ' + ui.cmdReason;
        return s;
      };
      scope.clipLine = function () {
        if (ui.link === 'none') return '--';
        return dash(ui.encodeBackend) + ' - last ' + dash(ui.clipTrigger);
      };
      scope.beatLine = function () {
        if (ui.link === 'none') return 'no gvd_state.json yet';
        var age = (ui.hbAge === null || ui.hbAge === undefined) ? '--' : Number(ui.hbAge).toFixed(2);
        return ui.link + ' - ' + age + ' s since last beat';
      };

      // --------------------------------------------------------------- feed
      function ingest(data) {
        for (var k in data) {
          if (Object.prototype.hasOwnProperty.call(data, k)) ui[k] = data[k];
        }
        if (data.link === 'none') {
          for (var i = 0; i < TELEMETRY.length; i++) ui[TELEMETRY[i]] = null;
        }
      }

      scope.$on('gvdUi', function (event, data) {
        if (!data) return;
        scope.$evalAsync(function () { ingest(data); });
      });
      // Older strip-only pushes still keep the engage state honest.
      scope.$on('gvdStrip', function (event, data) {
        if (!data || typeof data.engaged !== 'boolean') return;
        scope.$evalAsync(function () { ui.engaged = data.engaged; });
      });

      // Ask the extension for a first payload; it pushes on its own afterwards.
      var tries = 0;
      var poke = window.setInterval(function () {
        tries++;
        if (ui.link !== 'none' || tries > 10) { window.clearInterval(poke); poke = null; return; }
        lua('if extensions and extensions.gvd_main and extensions.gvd_main.pushUiState then extensions.gvd_main.pushUiState() end');
      }, 1000);

      scope.$on('$destroy', function () {
        if (poke) window.clearInterval(poke);
      });
    }
  };
}]);
