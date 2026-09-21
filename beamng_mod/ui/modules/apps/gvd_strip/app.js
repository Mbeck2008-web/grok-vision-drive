angular.module('beamng.apps')
.directive('gvdStrip', [function () {
  return {
    templateUrl: '/ui/modules/apps/gvd_strip/app.html',
    replace: true,
    restrict: 'EA',
    link: function (scope, element, attrs) {
      var el = element[0].querySelector('#gvdStripText');
      scope.$on('gvdStrip', function (event, data) {
        if (!el || !data) return;
        el.textContent = data.text || ('GVD  ' + (data.mode || '') + '  ' +
          Math.round(data.hz || 0) + 'Hz  TTC ' + (data.ttc == null ? '--' : data.ttc) +
          '  N=' + (data.n || 0));
        var tag = String(data.tag || '');
        var cls = 'gvd-strip';
        if (data.link === 'mismatch' || tag === 'MISMATCH') cls += ' is-mismatch';
        else if (tag === 'HOLD') cls += ' is-hold';
        else if (tag === 'DRIVE') cls += ' is-drive';
        else if (tag === 'ON' || data.engaged) cls += ' is-on';
        element[0].className = cls;
      });
    }
  };
}]);
