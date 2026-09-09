// GVD in-game app: engage + in-world viz toggles + a small ego-centric VISION scene.
// Data comes from the existing gvdUi guihooks push (gvd/main.lua reads Documents/GVD/gvd_state.json).
// The scene is a toy driving visualization drawn from our own corridor + tracked-object boxes.
// Titles stay GVD / VISION; scripts/test_gvd_ui_app.py fails the build on any other product chrome.
angular.module('beamng.apps')
.directive('gvdApp', [function () {
  return {
    templateUrl: '/ui/modules/apps/GVD/app.html',
    replace: true,
    restrict: 'EA',
    scope: true,
    link: function (scope, element) {
      var root = element[0];
      var POLICIES = ['modular', 'e2e', 'shadow'];

      // Same palette as the OpenCV GVD VISION window (python/viz/stage.py).
      var VOID = '#07080a';
      var ICE = [158, 196, 212];
      var CORRIDOR = [90, 167, 199];
      var PAPER = [232, 230, 225];
      var GHOST = [170, 200, 210];
      var PED = [215, 176, 138];
      var BIKE = [224, 184, 90];
      var EGO_BODY = [86, 98, 110];

      // Chase camera: ego centred at ~80% height, horizon at ~25%.
      var EYE = [0, -9.5, 4.70];
      var TARGET = [0, 14.95, -0.50];
      var VFOV = 46 * Math.PI / 180;
      var FPS_LIVE = 24;
      var FPS_IDLE = 6;
      var PATH_FADE_START = 25;
      var PATH_FADE_END = 42;

      var ui = {
        engaged: false, applying: false, link: 'none', hbAge: null, disengageReason: null,
        showPath: true, showGhosts: false, showScene: true,
        policy: null, policyReq: null, e2eOk: null, vetoReason: null, e2eBackend: null,
        hz: null, camHz: null, ttc: null, aeb: null, n: null, speed: null,
        pathConf: null, pathWidth: null, pathPreview: null, laneConf: null,
        camBackend: null, camNote: null, camOk: null, camTotal: null,
        vizWindow: null, vizScreen: null, vizNote: null, vizScreenReq: null,
        inferMs: null, rssMb: null, vramUsed: null, vramTotal: null, gpu: null,
        detector: null, actuator: null, cmdApplied: null, cmdReason: null,
        cmdSeq: null, cmdAckSeq: null, egoSource: null,
        clipTrigger: null, encodeBackend: null,
        path: null, tracks: null, lanes: null
      };
      // Cleared when the supervisor is gone: stale numbers must not look live.
      var TELEMETRY = [
        'hz', 'camHz', 'ttc', 'aeb', 'n', 'speed', 'pathConf', 'pathWidth', 'pathPreview',
        'laneConf', 'policy', 'e2eOk', 'vetoReason', 'e2eBackend', 'camBackend', 'camNote',
        'camOk', 'camTotal', 'vizWindow', 'vizScreen', 'vizNote', 'inferMs', 'rssMb',
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
      scope.setScene = function (v) { ui.showScene = !!v; gvdLua('setShowScene(' + (v ? 'true' : 'false') + ')'); };
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
      function dash(v) { return (v === null || v === undefined || v === '') ? '—' : v; }
      scope.dash = dash;
      scope.fmt1 = function (v) { return (v === null || v === undefined) ? '--' : Number(v).toFixed(1); };
      scope.kph = function () {
        if (ui.speed === null || ui.speed === undefined || ui.link !== 'live') return '--';
        return String(Math.round(Number(ui.speed) * 3.6));
      };

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
          if (driving()) return 'GVD holds the wheel · steer to take over';
          if (ui.actuator === 'cmd_json') return 'armed · waiting for supervisor commands';
          return 'armed · actuators idle';
        }
        var r = ui.disengageReason;
        if (r && r !== 'none' && r !== 'not_engaged') return 'last: ' + r;
        return 'standby';
      };
      scope.veil = function () {
        if (ui.link === 'none') return 'no telemetry · start the GVD python stack';
        var age = (ui.hbAge === null || ui.hbAge === undefined) ? null : Number(ui.hbAge).toFixed(1) + ' s';
        return 'telemetry stale' + (age ? ' · ' + age : '');
      };

      scope.pathLine = function () {
        if (ui.link === 'none') return 'waiting for gvd_state.json';
        var low = ui.pathConf !== null && ui.pathConf < 0.45;
        var bits = ['conf ' + (ui.pathConf === null ? '--' : Number(ui.pathConf).toFixed(2)) + (low ? ' low' : '')];
        if (ui.pathWidth !== null && ui.pathWidth !== undefined) bits.push(Number(ui.pathWidth).toFixed(1) + ' m');
        if (ui.pathPreview) bits.push('preview path');
        if (ui.n !== null && ui.n !== undefined) bits.push(ui.n + ' tracked');
        return bits.join(' · ');
      };
      scope.segClass = function (p) {
        return { 'is-on': ui.policy === p, 'is-req': ui.policyReq === p && ui.policy !== p };
      };
      scope.policyLine = function () {
        if (ui.link === 'none') {
          return ui.policyReq ? ('requested ' + ui.policyReq + ' · supervisor not running') : 'supervisor not running';
        }
        var bits = ['e2e ' + dash(ui.e2eBackend)];
        bits.push('veto ' + dash(ui.vetoReason));
        if (ui.e2eOk === false) bits.push('e2e held by modular');
        if (ui.policyReq && ui.policyReq !== ui.policy) bits.push('requested ' + ui.policyReq + '…');
        return bits.join(' · ');
      };
      scope.camLine = function () {
        if (ui.link === 'none') return '—';
        var s = dash(ui.camBackend);
        if (ui.camTotal) s += ' · ' + ui.camOk + '/' + ui.camTotal + ' feeds';
        if (ui.camNote) s += ' · ' + ui.camNote;
        else if (ui.camBackend === 'window') s += ' · retail: cam_main only';
        return s;
      };
      scope.vizLine = function () {
        if (ui.link === 'none') return '—';
        if (!ui.vizWindow) return 'off · run python with --viz';
        return 'on · ' + dash(ui.vizNote || ('screen ' + dash(ui.vizScreen)));
      };
      scope.vizSel = function (s) {
        if (ui.vizScreenReq) return ui.vizScreenReq === s;
        return String(ui.vizScreen) === s;
      };
      scope.rateLine = function () {
        if (ui.link === 'none') return '—';
        return Math.round(ui.hz || 0) + ' Hz loop · ' + Math.round(ui.camHz || 0) + ' Hz cams · ' +
          (ui.inferMs === null ? '--' : Number(ui.inferMs).toFixed(1)) + ' ms infer';
      };
      scope.gpuLine = function () {
        if (ui.link === 'none') return '—';
        var vram = (ui.vramUsed === null ? '--' : Number(ui.vramUsed).toFixed(1)) + ' / ' +
          (ui.vramTotal === null ? '--' : Number(ui.vramTotal).toFixed(1)) + ' GB';
        return vram + (ui.gpu ? ' · ' + ui.gpu : '') + (ui.rssMb ? ' · rss ' + Math.round(ui.rssMb) + ' MB' : '');
      };
      scope.actuatorLine = function () {
        if (ui.link === 'none') return '—';
        var s = dash(ui.actuator);
        if (ui.applying) s += ' · mod driving';
        else if (ui.cmdApplied === false) s += ' · not applied';
        if (ui.cmdAckSeq != null && ui.cmdAckSeq >= 0) s += ' · ack ' + ui.cmdAckSeq + '/' + dash(ui.cmdSeq);
        if (ui.egoSource && ui.egoSource !== 'none') s += ' · ego ' + ui.egoSource;
        if (ui.cmdReason) s += ' · ' + ui.cmdReason;
        return s;
      };
      scope.clipLine = function () {
        if (ui.link === 'none') return '—';
        return dash(ui.encodeBackend) + ' · last ' + dash(ui.clipTrigger);
      };
      scope.beatLine = function () {
        if (ui.link === 'none') return 'no gvd_state.json yet';
        var age = (ui.hbAge === null || ui.hbAge === undefined) ? '--' : Number(ui.hbAge).toFixed(2);
        return ui.link + ' · ' + age + ' s since last beat';
      };

      // --------------------------------------------------------------- feed
      function ingest(data) {
        for (var k in data) {
          if (Object.prototype.hasOwnProperty.call(data, k)) ui[k] = data[k];
        }
        // absence is meaningful for geometry (toggles off / no perception yet)
        ui.path = data.path || null;
        ui.tracks = data.tracks || null;
        ui.lanes = data.lanes || null;
        if (data.link === 'none') {
          for (var i = 0; i < TELEMETRY.length; i++) ui[TELEMETRY[i]] = null;
        }
        sceneDirty = true;
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

      // ------------------------------------------------------------- scene
      function rgba(c, a) { return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')'; }
      function shade(c, k) {
        return [Math.round(c[0] * k), Math.round(c[1] * k), Math.round(c[2] * k)];
      }
      function norm3(v) {
        var l = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) || 1e-9;
        return [v[0] / l, v[1] / l, v[2] / l];
      }
      function cross3(a, b) {
        return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
      }
      function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

      var fwd = norm3([TARGET[0] - EYE[0], TARGET[1] - EYE[1], TARGET[2] - EYE[2]]);
      var rightV = norm3(cross3(fwd, [0, 0, 1]));
      var upV = cross3(rightV, fwd);
      var cw = 0, ch = 0, fl = 0, horizonY = 0;

      function sizeChanged(w, h) {
        cw = w; ch = h;
        fl = (h * 0.5) / Math.tan(VFOV * 0.5);
        var d = [0, 1, 0];
        horizonY = h * 0.5 - fl * (dot3(d, upV) / dot3(d, fwd));
      }
      // project ego-frame (x right, y forward, z up) → canvas px
      function P(x, y, z) {
        var e = [x - EYE[0], y - EYE[1], (z || 0) - EYE[2]];
        var cz = dot3(e, fwd);
        if (cz < 0.35) cz = 0.35;
        return [cw * 0.5 + fl * dot3(e, rightV) / cz, ch * 0.5 - fl * dot3(e, upV) / cz, cz];
      }
      function poly(ctx, pts) {
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (var i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
        ctx.closePath();
      }

      function drawGround(ctx) {
        var g = ctx.createLinearGradient(0, horizonY, 0, ch);
        g.addColorStop(0, 'rgba(16,21,26,0)');
        g.addColorStop(0.35, 'rgba(20,26,32,0.9)');
        g.addColorStop(1, 'rgba(27,34,41,1)');
        poly(ctx, [P(-16, 60, 0), P(16, 60, 0), P(16, -12, 0), P(-16, -12, 0)]);
        ctx.fillStyle = g;
        ctx.fill();
        ctx.strokeStyle = 'rgba(158,196,212,0.05)';
        ctx.lineWidth = 1;
        for (var d = 10; d <= 30; d += 10) {
          var a = P(-4.5, d, 0), b = P(4.5, d, 0);
          ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
        }
      }

      function drawLanes(ctx, lanes) {
        if (!lanes || !lanes.length) return;
        for (var i = 0; i < lanes.length; i++) {
          var pl = lanes[i];
          if (!pl || pl.length < 2) continue;
          ctx.beginPath();
          for (var j = 0; j < pl.length; j++) {
            var p = P(pl[j].x, pl[j].y, 0.02);
            if (j === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
          }
          ctx.strokeStyle = rgba(PAPER, 0.10); ctx.lineWidth = 5; ctx.stroke();
          ctx.strokeStyle = rgba(PAPER, 0.42); ctx.lineWidth = 1.4; ctx.stroke();
        }
      }

      function drawCorridor(ctx, path, halfW, conf, preview) {
        if (!path || path.length < 2) return;
        var left = [], right = [], mid = [], fade = [];
        var dist = 0;
        for (var i = 0; i < path.length; i++) {
          var p = path[i];
          var q = path[i + 1] || path[i - 1];
          var tx = (path[i + 1] ? q.x - p.x : p.x - q.x);
          var ty = (path[i + 1] ? q.y - p.y : p.y - q.y);
          var l = Math.sqrt(tx * tx + ty * ty) || 1e-6;
          if (i > 0) {
            var dx = p.x - path[i - 1].x, dy = p.y - path[i - 1].y;
            dist += Math.sqrt(dx * dx + dy * dy);
          }
          if (dist > PATH_FADE_END) break;
          var nx = ty / l * halfW, ny = -tx / l * halfW;
          left.push(P(p.x - nx, p.y - ny, 0.03));
          right.push(P(p.x + nx, p.y + ny, 0.03));
          mid.push(P(p.x, p.y, 0.03));
          fade.push(dist <= PATH_FADE_START ? 1 : Math.max(0, 1 - (dist - PATH_FADE_START) / (PATH_FADE_END - PATH_FADE_START)));
        }
        for (var s = 0; s < left.length - 1; s++) {
          var a = 0.52 * Math.max(0.2, conf) * (0.55 * fade[s] + 0.45 * fade[s + 1]);
          if (a < 0.02) continue;
          poly(ctx, [left[s], left[s + 1], right[s + 1], right[s]]);
          ctx.fillStyle = rgba(CORRIDOR, a);
          ctx.fill();
        }
        ctx.setLineDash(preview ? [5, 4] : []);
        for (var e = 0; e < left.length - 1; e++) {
          ctx.strokeStyle = rgba(ICE, 0.7 * fade[e]);
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.moveTo(left[e][0], left[e][1]); ctx.lineTo(left[e + 1][0], left[e + 1][1]);
          ctx.moveTo(right[e][0], right[e][1]); ctx.lineTo(right[e + 1][0], right[e + 1][1]);
          ctx.stroke();
        }
        ctx.setLineDash([4, 5]);
        ctx.strokeStyle = rgba(ICE, 0.30);
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (var m = 0; m < mid.length; m++) {
          if (m === 0) ctx.moveTo(mid[m][0], mid[m][1]); else ctx.lineTo(mid[m][0], mid[m][1]);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // yaw is the ego-frame heading angle from +X (π/2 = straight ahead), as written by the tracker.
      function drawBox(ctx, x, y, yaw, L, W, H, col, alpha, strokeCol, strokeW) {
        var hx = Math.cos(yaw), hy = Math.sin(yaw);
        var sx = hy, sy = -hx;
        var hl = L * 0.5, hw = W * 0.5;
        var base = [];
        var top = [];
        var shadow = [];
        var signs = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
        for (var i = 0; i < 4; i++) {
          var px = x + hx * hl * signs[i][0] + sx * hw * signs[i][1];
          var py = y + hy * hl * signs[i][0] + sy * hw * signs[i][1];
          base.push(P(px, py, 0));
          top.push(P(px, py, H));
          shadow.push(P(x + (px - x) * 1.18, y + (py - y) * 1.18, 0));
        }
        poly(ctx, shadow);
        ctx.fillStyle = 'rgba(0,0,0,' + (0.42 * alpha).toFixed(3) + ')';
        ctx.fill();

        // convex box: keep only faces whose outward normal points at the camera, far ones first
        var faces = [
          { p: [top[0], top[1], top[2], top[3]], n: [0, 0, 1], c: [x, y, H], k: 1.0 },
          { p: [base[1], base[2], top[2], top[1]], n: [hx, hy, 0], c: [x + hx * hl, y + hy * hl, H * 0.5], k: 0.62 },
          { p: [base[3], base[0], top[0], top[3]], n: [-hx, -hy, 0], c: [x - hx * hl, y - hy * hl, H * 0.5], k: 0.5 },
          { p: [base[2], base[3], top[3], top[2]], n: [sx, sy, 0], c: [x + sx * hw, y + sy * hw, H * 0.5], k: 0.4 },
          { p: [base[0], base[1], top[1], top[0]], n: [-sx, -sy, 0], c: [x - sx * hw, y - sy * hw, H * 0.5], k: 0.4 }
        ];
        var vis = [];
        for (var f = 0; f < faces.length; f++) {
          var fc = faces[f];
          var v = [fc.c[0] - EYE[0], fc.c[1] - EYE[1], fc.c[2] - EYE[2]];
          if (dot3(fc.n, v) >= 0) continue;
          fc.d = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
          vis.push(fc);
        }
        vis.sort(function (a, b) { return b.d - a.d; });
        for (var g = 0; g < vis.length; g++) {
          poly(ctx, vis[g].p);
          ctx.fillStyle = rgba(shade(col, vis[g].k), alpha);
          ctx.fill();
          if (strokeCol) {
            ctx.strokeStyle = rgba(strokeCol, Math.min(1, alpha * 0.55));
            ctx.lineWidth = strokeW || 1;
            ctx.stroke();
          }
        }
      }

      function trackDims(cls) {
        if (cls === 'pedestrian' || cls === 'ped') return [0.6, 0.6, 1.75, PED];
        if (cls === 'bicycle' || cls === 'bike') return [1.8, 0.6, 1.6, BIKE];
        return [4.2, 1.8, 1.5, GHOST];
      }

      function drawTracks(ctx, tracks) {
        if (!tracks || !tracks.length) return;
        var sorted = tracks.slice().sort(function (a, b) { return (b.y || 0) - (a.y || 0); });
        for (var i = 0; i < sorted.length; i++) {
          var t = sorted[i];
          var d = trackDims(t.cls);
          var far = Math.max(0.18, Math.min(1, 1 - ((t.y || 0) - 30) / 25));
          var lead = !!t.lead;
          drawBox(ctx, t.x || 0, t.y || 0, (t.yaw === undefined || t.yaw === null) ? Math.PI / 2 : t.yaw,
            d[0], d[1], d[2], lead ? ICE : d[3], (lead ? 0.72 : 0.5) * far * (t.a === undefined ? 1 : t.a),
            lead ? ICE : PAPER, lead ? 1.4 : 0.7);
          if (lead) {
            var tag = P(t.x || 0, t.y || 0, d[2] + 0.55);
            ctx.fillStyle = rgba(ICE, 0.9);
            ctx.font = '9px Consolas, monospace';
            ctx.textAlign = 'center';
            ctx.fillText('LEAD', tag[0], tag[1]);
            ctx.textAlign = 'left';
          }
        }
      }

      function drawEgo(ctx, engaged) {
        if (engaged) {
          var c = P(0, 0.2, 0.02);
          var r = Math.abs(P(2.6, 0.2, 0.02)[0] - c[0]);
          var g = ctx.createRadialGradient(c[0], c[1], 1, c[0], c[1], Math.max(6, r));
          g.addColorStop(0, rgba(ICE, 0.30));
          g.addColorStop(1, rgba(ICE, 0));
          ctx.fillStyle = g;
          ctx.beginPath(); ctx.arc(c[0], c[1], Math.max(6, r), 0, Math.PI * 2); ctx.fill();
        }
        drawBox(ctx, 0, 0, Math.PI / 2, 4.4, 1.85, 0.78, EGO_BODY, 1, [128, 142, 156], 0.8);
        drawBox(ctx, 0, -0.25, Math.PI / 2, 2.2, 1.58, 1.32, shade(EGO_BODY, 1.22), 1, [128, 142, 156], 0.8);
      }

      // -------------------------------------------------- smoothing + frame
      var smoothTracks = {};
      var smoothPath = null;
      var sceneDirty = true;

      function lerp(a, b, k) { return a + (b - a) * k; }

      function smooth(dt) {
        var k = 1 - Math.exp(-dt / 0.10);
        var seen = {};
        var list = angular.isArray(ui.tracks) ? ui.tracks : [];
        for (var i = 0; i < list.length; i++) {
          var t = list[i];
          var id = (t.id === undefined || t.id === null) ? ('i' + i) : ('t' + t.id);
          seen[id] = true;
          var s = smoothTracks[id];
          if (!s) {
            s = { x: t.x || 0, y: t.y || 0, yaw: t.yaw, a: 0, cls: t.cls, lead: t.lead };
            smoothTracks[id] = s;
          }
          s.x = lerp(s.x, t.x || 0, k);
          s.y = lerp(s.y, t.y || 0, k);
          // interpolate heading through its unit vector so ±π never spins the box
          var ty = (t.yaw === undefined || t.yaw === null) ? Math.PI / 2 : t.yaw;
          var vx = lerp(Math.cos(s.yaw), Math.cos(ty), k);
          var vy = lerp(Math.sin(s.yaw), Math.sin(ty), k);
          s.yaw = Math.atan2(vy, vx);
          s.a = Math.min(1, s.a + dt * 4);
          s.cls = t.cls; s.lead = t.lead;
        }
        var out = [];
        for (var id2 in smoothTracks) {
          if (!Object.prototype.hasOwnProperty.call(smoothTracks, id2)) continue;
          var st = smoothTracks[id2];
          if (!seen[id2]) {
            st.a -= dt * 3;
            if (st.a <= 0.02) { delete smoothTracks[id2]; continue; }
          }
          out.push(st);
        }

        var path = angular.isArray(ui.path) ? ui.path : null;
        if (!path) {
          smoothPath = null;
        } else if (!smoothPath || smoothPath.length !== path.length) {
          smoothPath = path.map(function (p) { return { x: p.x, y: p.y }; });
        } else {
          for (var j = 0; j < path.length; j++) {
            smoothPath[j].x = lerp(smoothPath[j].x, path[j].x, k);
            smoothPath[j].y = lerp(smoothPath[j].y, path[j].y, k);
          }
        }
        return out;
      }

      var canvas = null;
      var raf = null;
      var lastFrame = 0;

      function getCanvas() {
        if (canvas && canvas.isConnected !== false && canvas.parentNode) return canvas;
        canvas = root.querySelector('.gvd-canvas');
        return canvas;
      }

      function frame(ts) {
        raf = window.requestAnimationFrame(frame);
        var cv = getCanvas();
        if (!cv || !ui.showScene || root.offsetParent === null) return;
        var live = ui.link === 'live';
        var budget = 1000 / (live ? FPS_LIVE : FPS_IDLE);
        var dt = (ts - lastFrame) / 1000;
        if (ts - lastFrame < budget) return;
        lastFrame = ts;
        if (dt > 0.5) dt = 0.5;

        var w = cv.clientWidth, h = cv.clientHeight;
        if (!w || !h) return;
        var dpr = window.devicePixelRatio || 1;
        if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
          cv.width = Math.round(w * dpr);
          cv.height = Math.round(h * dpr);
        }
        var ctx = cv.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        if (w !== cw || h !== ch) sizeChanged(w, h);

        var tracks = smooth(dt);
        ctx.fillStyle = VOID;
        ctx.fillRect(0, 0, w, h);
        ctx.globalAlpha = ui.link === 'stale' ? 0.55 : 1;
        drawGround(ctx);
        drawLanes(ctx, ui.lanes);
        if (smoothPath) {
          drawCorridor(ctx, smoothPath, Math.max(0.9, (ui.pathWidth || 2.0) * 0.5),
            ui.pathConf === null || ui.pathConf === undefined ? 0.5 : ui.pathConf, !!ui.pathPreview);
        }
        drawTracks(ctx, tracks);
        drawEgo(ctx, ui.engaged && live);
        ctx.globalAlpha = 1;
      }

      raf = window.requestAnimationFrame(frame);

      // Ask the extension for a first payload; it pushes on its own afterwards.
      var tries = 0;
      var poke = window.setInterval(function () {
        tries++;
        if (ui.link !== 'none' || tries > 10) { window.clearInterval(poke); poke = null; return; }
        lua('if extensions and extensions.gvd_main and extensions.gvd_main.pushUiState then extensions.gvd_main.pushUiState() end');
      }, 1000);

      scope.$on('$destroy', function () {
        if (raf) window.cancelAnimationFrame(raf);
        if (poke) window.clearInterval(poke);
      });
    }
  };
}]);
