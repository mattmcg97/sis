<script>
// Every panel folds. Done here rather than in the markup so a panel does
// not have to know it is foldable -- the heading becomes the handle and
// everything after it becomes the body.
(function () {
  var panels = document.querySelectorAll('section.panel');
  Array.prototype.forEach.call(panels, function (panel) {
    var heading = panel.querySelector(':scope > h2');
    if (!heading) return;
    var body = document.createElement('div');
    body.className = 'fold-body';
    while (heading.nextSibling) body.appendChild(heading.nextSibling);
    panel.appendChild(body);
    panel.classList.add('foldable');
    heading.setAttribute('role', 'button');
    heading.setAttribute('tabindex', '0');
    heading.addEventListener('click', function () {
      panel.classList.toggle('folded');
    });
    heading.addEventListener('keydown', function (event) {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      event.preventDefault();
      panel.classList.toggle('folded');
    });
  });
  var button = document.getElementById('foldAll');
  if (!button) return;
  button.addEventListener('click', function () {
    var foldable = document.querySelectorAll('section.panel.foldable');
    var collapsing = button.textContent.indexOf('collapse') === 0;
    Array.prototype.forEach.call(foldable, function (panel) {
      panel.classList.toggle('folded', collapsing);
    });
    Array.prototype.forEach.call(document.querySelectorAll('details.panel'),
      function (node) { node.open = !collapsing; });
    button.textContent = collapsing ? 'expand all' : 'collapse all';
  });
})();

Array.prototype.forEach.call(document.querySelectorAll('table.sortable'), function (table) {
  var headers = table.tHead.rows[0].cells;
  var body = table.tBodies[0];
  Array.prototype.forEach.call(headers, function (header, index) {
    header.addEventListener('click', function () {
      var descending = !header.classList.contains('desc');
      Array.prototype.forEach.call(headers, function (other) {
        other.classList.remove('asc', 'desc');
      });
      header.classList.add(descending ? 'desc' : 'asc');
      var rows = Array.prototype.slice.call(body.rows);
      rows.sort(function (a, b) {
        var x = a.cells[index], y = b.cells[index];
        var xv = x.dataset.v !== undefined ? x.dataset.v : x.textContent.trim();
        var yv = y.dataset.v !== undefined ? y.dataset.v : y.textContent.trim();
        var xn = parseFloat(xv), yn = parseFloat(yv);
        var numeric = !isNaN(xn) && !isNaN(yn);
        if (numeric) return descending ? yn - xn : xn - yn;
        // Blanks last either way.
        if (xv === '') return 1;
        if (yv === '') return -1;
        return descending ? yv.localeCompare(xv) : xv.localeCompare(yv);
      });
      var fragment = document.createDocumentFragment();
      rows.forEach(function (row) { fragment.appendChild(row); });
      body.appendChild(fragment);
    });
  });
});

// Click any row to pin it, click again to unpin, Escape to clear. Survives
// sorting and filtering because the class rides on the row element itself.
(function () {
  document.addEventListener('click', function (event) {
    var cell = event.target.closest('td, th');
    if (!cell) return;
    var row = cell.parentElement;
    if (!row || row.parentElement.tagName !== 'TBODY') return;
    // Selecting text inside a row should not also pin it.
    if (String(window.getSelection())) return;
    row.classList.toggle('picked');
  });
  document.addEventListener('keydown', function (event) {
    if (event.key !== 'Escape') return;
    var picked = document.querySelectorAll('tbody tr.picked');
    for (var i = 0; i < picked.length; i++) picked[i].classList.remove('picked');
  });
})();

// The pair table is drawn from compact rows (#pairData) by column kind
// (#pairSpec): sorting and the match filter work on the data, and only the
// first LIMIT rows of the current view are put in the page at once.
(function () {
  var specNode = document.getElementById('pairSpec');
  var dataNode = document.getElementById('pairData');
  var table = document.getElementById('pairTable');
  if (!specNode || !dataNode || !table) return;
  var spec = JSON.parse(specNode.textContent);
  var rows = JSON.parse(dataNode.textContent);
  var kinds = spec.kinds, groups = spec.groups;
  var input = document.getElementById('pairFilter');
  var label = document.getElementById('pairCount');
  var body = table.tBodies[0];
  var LIMIT = 2000;
  var view = rows.slice();
  var total = rows.length;

  function gap(value, cuts) {
    var m = Math.abs(value), i = 0;
    while (i < cuts.length && m > cuts[i]) i++;
    return 'g' + i;
  }
  function fixed(v, n) { return v.toFixed(n); }
  function signed(v, n) { return (v > 0 ? '+' : '') + v.toFixed(n); }
  function dash(td) { td.innerHTML = '&mdash;'; }

  function cell(kind, v, td) {
    if (v === null || v === undefined || v === '') {
      if (kind === 'dd' || kind === 'i' || kind === 'l' || kind === 'p4' || kind === 'dp' ||
          kind === 'te' || kind === 'b' || kind === 'o' || kind === 'c') dash(td);
      return;
    }
    switch (kind) {
      case 't': td.textContent = v; break;
      case 'i': td.textContent = v; break;
      case 'sd': td.textContent = signed(v, 0); break;
      case 'mg': td.textContent = signed(v, 0); td.className = gap(v, spec.message_gap); break;
      case 'l': td.textContent = signed(v, 1); break;
      case 'p4': td.textContent = fixed(v, 4); break;
      case 'dp': td.textContent = signed(v, 4); td.className = gap(v, spec.prob_delta); break;
      case 'b': td.innerHTML = '<b>' + v + '</b>'; break;
      case 'te': td.textContent = v.toLocaleString(); if (v <= 120) td.className = 'warn'; break;
      case 'dd': td.textContent = v[0]; if (v[2]) td.className = 'warn'; break;
      case 'o':
        td.innerHTML = v ? '<span class="good">won</span>' : '<span class="bad">lost</span>';
        break;
      case 'c':
        var span = document.createElement('span');
        span.className = v[1] || 'dim';
        span.textContent = v[0];
        td.appendChild(span);
        break;
      case 'live': td.textContent = v; td.className = v === 'live' ? 'dim' : 'bad'; break;
      default: td.textContent = v;
    }
  }

  function draw() {
    var fragment = document.createDocumentFragment();
    var n = Math.min(view.length, LIMIT);
    for (var r = 0; r < n; r++) {
      var row = view[r];
      var tr = document.createElement('tr');
      if (row[row.length - 1] !== 'live') tr.className = 'notlive';
      for (var c = 0; c < kinds.length; c++) {
        var td = document.createElement('td');
        cell(kinds[c], row[c], td);
        if (groups[c]) td.className = (td.className + ' grp').trim();
        tr.appendChild(td);
      }
      fragment.appendChild(tr);
    }
    body.innerHTML = '';
    body.appendChild(fragment);
    label.textContent = (view.length > n ? 'first ' + n.toLocaleString() + ' of ' : '') +
      view.length.toLocaleString() + ' rows' +
      (view.length === total ? '' : ' \u00b7 ' + total.toLocaleString() + ' in all');
  }

  function sortKey(kind, v) {
    if (v === null || v === undefined) return null;
    if (kind === 'dd') return v[1] === '' ? null : v[1];
    if (kind === 'c') return v[0];
    if (kind === 'dp' || kind === 'mg') return Math.abs(v);
    return v;
  }

  var sortedBy = -1, descending = false;
  var headers = table.tHead.rows[0].cells;
  Array.prototype.forEach.call(headers, function (header, index) {
    header.style.cursor = 'pointer';
    header.addEventListener('click', function () {
      descending = sortedBy === index ? !descending : true;
      sortedBy = index;
      Array.prototype.forEach.call(headers, function (h) { h.classList.remove('asc', 'desc'); });
      header.classList.add(descending ? 'desc' : 'asc');
      var kind = kinds[index];
      view.sort(function (a, b) {
        var x = sortKey(kind, a[index]), y = sortKey(kind, b[index]);
        if (x === null) return 1;          // blanks last either way
        if (y === null) return -1;
        if (typeof x === 'number' && typeof y === 'number') return descending ? y - x : x - y;
        return descending ? String(y).localeCompare(String(x)) : String(x).localeCompare(String(y));
      });
      draw();
    });
  });

  var pending = null;
  function apply() {
    var needle = input.value.trim().toLowerCase();
    view = needle ? rows.filter(function (row) {
      return String(row[1]).toLowerCase().indexOf(needle) !== -1;
    }) : rows.slice();
    sortedBy = -1;
    Array.prototype.forEach.call(headers, function (h) { h.classList.remove('asc', 'desc'); });
    draw();
  }
  if (input) {
    input.addEventListener('input', function () {
      if (pending) clearTimeout(pending);
      pending = setTimeout(apply, 120);
    });
    input.addEventListener('search', apply);
  }
  draw();
})();
</script>