document.addEventListener('DOMContentLoaded', function () {
  // Navbar burger toggle (mobile).
  document.querySelectorAll('.navbar-burger').forEach(function (burger) {
    burger.addEventListener('click', function () {
      burger.classList.toggle('is-active');
      document.querySelectorAll('.navbar-menu').forEach(function (m) { m.classList.toggle('is-active'); });
    });
  });

  // Leaderboard: click a numeric header to sort (descending first), click "Paradigm" to restore paper order.
  var table = document.getElementById('leaderboard');
  if (!table) return;
  var tbody = table.tBodies[0];
  var headers = table.tHead.rows[0].cells;
  var state = { col: null, desc: true };

  function sortBy(colIdx, mode) {
    var rows = Array.prototype.slice.call(tbody.rows);
    if (mode === 'paper') {
      rows.sort(function (a, b) { return +a.dataset.order - +b.dataset.order; });
      state = { col: null, desc: true };
    } else {
      state.desc = state.col === colIdx ? !state.desc : true;
      state.col = colIdx;
      rows.sort(function (a, b) {
        var va = parseFloat(a.cells[colIdx].dataset.v), vb = parseFloat(b.cells[colIdx].dataset.v);
        if (isNaN(va) || isNaN(vb)) throw new Error('Leaderboard cell without numeric data-v in column ' + colIdx);
        return state.desc ? vb - va : va - vb;
      });
    }
    rows.forEach(function (r) { tbody.appendChild(r); });
    Array.prototype.forEach.call(headers, function (h, i) {
      h.classList.remove('sorted-asc', 'sorted-desc');
      if (i === state.col) h.classList.add(state.desc ? 'sorted-desc' : 'sorted-asc');
    });
  }

  Array.prototype.forEach.call(headers, function (h, i) {
    var mode = h.dataset.sort;
    if (!mode) return;
    h.classList.add('sortable');
    h.addEventListener('click', function () { sortBy(i, mode); });
  });
});
