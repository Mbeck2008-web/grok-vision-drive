angular.module('beamng.apps')
.directive('gvdApp', [function () {
  return {
    templateUrl: '/ui/modules/apps/GVD/app.html',
    replace: true,
    restrict: 'EA',
    link: function (scope, element, attrs) {
      var root = element[0];
      var statusEl = root.querySelector('#gvdUiStatus');
      var btn = root.querySelector('#gvdBtnEngage');
      var chkPath = root.querySelector('#gvdChkPath');
      var chkGhosts = root.querySelector('#gvdChkGhosts');
      var engaged = false;
      var luaFailLogged = false;

      function lua(cmd) {
        try {
          if (typeof bngApi !== 'undefined' && bngApi.engineLua) {
            bngApi.engineLua(cmd);
            return true;
          }
        } catch (e) {
          if (!luaFailLogged) {
            luaFailLogged = true;
            try { console.warn('[GVD] engineLua failed', e); } catch (e2) {}
          }
          return false;
        }
        if (!luaFailLogged) {
          luaFailLogged = true;
          try { console.warn('[GVD] bngApi.engineLua unavailable'); } catch (e2) {}
        }
        return false;
      }

      function paintEngage() {
        if (!btn) return;
        btn.textContent = engaged ? 'Disengage' : 'Engage';
        btn.classList.toggle('gvd-on', !!engaged);
      }

      function onUi(data) {
        if (!data) return;
        if (typeof data.engaged === 'boolean') {
          engaged = data.engaged;
          paintEngage();
        }
        if (chkPath && typeof data.showPath === 'boolean') {
          chkPath.checked = data.showPath;
        }
        if (chkGhosts && typeof data.showGhosts === 'boolean') {
          chkGhosts.checked = data.showGhosts;
        }
        if (statusEl) {
          var ttc = (data.ttc == null || data.ttc === undefined) ? '--' : Number(data.ttc).toFixed(1);
          var mode = engaged ? (data.applying ? 'DRIVE' : 'ON') : 'OFF';
          statusEl.textContent = mode + ' · ' +
            Math.round(data.hz || 0) + 'Hz · TTC ' + ttc + ' · N=' + (data.n || 0);
        }
      }

      scope.$on('gvdUi', function (event, data) { onUi(data); });
      scope.$on('gvdStrip', function (event, data) {
        if (!data) return;
        if (typeof data.engaged === 'boolean') {
          engaged = data.engaged;
          paintEngage();
        }
        if (statusEl && data.text) statusEl.textContent = data.text;
      });

      if (btn) {
        btn.addEventListener('click', function () {
          lua("extensions.gvd_main.toggleEngage()");
        });
      }
      if (chkPath) {
        chkPath.addEventListener('change', function () {
          lua("extensions.gvd_main.setShowPath(" + (chkPath.checked ? "true" : "false") + ")");
        });
      }
      if (chkGhosts) {
        chkGhosts.addEventListener('change', function () {
          lua("extensions.gvd_main.setShowAgentGhosts(" + (chkGhosts.checked ? "true" : "false") + ")");
        });
      }

      paintEngage();
      lua("if extensions.gvd_main then guihooks.trigger('gvdUi', {engaged=extensions.gvd_main.isEngaged(), showPath=(extensions.gvd_main.getShowPath and extensions.gvd_main.getShowPath()) or true, showGhosts=(extensions.gvd_main.getShowAgentGhosts and extensions.gvd_main.getShowAgentGhosts()) or false, hz=0, n=0}) end");
    }
  };
}]);
