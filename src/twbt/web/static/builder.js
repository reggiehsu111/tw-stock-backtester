/* Signal builder: a drag-and-drop block canvas and an expression editor,
   kept in sync where they can be.

   The two views are not equivalent and the page says so rather than pretending.
   A chain of blocks is always expressible as code, so Blocks -> Code is total.
   Code -> Blocks only works when the expression is a chain; a tree like
   Div(a, b) has two inputs and no place on a one-input canvas, so the blocks
   tab reports that instead of silently dropping a branch. */
(function () {
  'use strict';

  var PRESETS = JSON.parse(document.getElementById('presetdata').textContent);
  var CFG = JSON.parse(document.getElementById('cfg').textContent);
  var S = CFG.strings;
  var steps = [];            // [{op, params:{}}]
  var mode = 'blocks';       // 'blocks' | 'code'
  var dragFrom = null;       // index when reordering, null when from palette
  var dragOp = null;         // {op, params} when dragging from the palette

  function $(id) { return document.getElementById(id); }

  /* Hover help. A native title= attribute works but takes about a second to
     appear and cannot show the signature on its own line, which is the half
     people actually need when deciding what to type. */
  var tip = document.getElementById('tip');

  function showTip(el) {
    var hint = el.dataset.hint, sig = el.dataset.sig;
    if (!hint && !sig) return;
    tip.innerHTML = '';
    if (sig) {
      var c = document.createElement('code');
      c.className = 'tsig'; c.textContent = sig; tip.appendChild(c);
    }
    if (hint) tip.appendChild(document.createTextNode(hint));
    tip.hidden = false;
    var r = el.getBoundingClientRect(), t = tip.getBoundingClientRect();
    var left = Math.min(r.right + 10, window.innerWidth - t.width - 10);
    if (r.right + 10 + t.width > window.innerWidth) left = Math.max(10, r.left - t.width - 10);
    tip.style.left = left + 'px';
    tip.style.top = Math.max(10, Math.min(r.top, window.innerHeight - t.height - 10)) + 'px';
  }

  function hideTip() { tip.hidden = true; }

  document.addEventListener('mouseover', function (e) {
    var el = e.target.closest('[data-hint]');
    if (el) showTip(el); else hideTip();
  });
  document.addEventListener('mouseout', function (e) {
    if (e.target.closest('[data-hint]')) hideTip();
  });
  document.addEventListener('scroll', hideTip, true);
  window.addEventListener('blur', hideTip);
  var canvas = $('canvas'), expr = $('expr'), results = $('results'),
      status = $('status'), code = $('code'), codestatus = $('codestatus');

  /* ---------------- expression from blocks ---------------- */

  var MATRIX_FIRST = { GrossScale: 1, Pow: 1 };

  function chainExpression() {
    var src = $('source').value;
    var e = src === 'close' ? 'px' : (src === 'returns' ? 'Returns(px)' : src);
    steps.forEach(function (s) {
      var nums = Object.keys(s.params || {}).map(function (k) { return s.params[k]; });
      var args = MATRIX_FIRST[s.op] ? [e].concat(nums) : nums.concat([e]);
      e = s.op + '(' + args.join(', ') + ')';
    });
    return e;
  }

  function currentExpression() {
    return mode === 'code' ? code.value.trim() : chainExpression();
  }

  function refreshExpr() {
    expr.textContent = 'sig = ' + (currentExpression() || '…');
  }

  /* ---------------- block canvas ---------------- */

  function makeSlot(index) {
    var slot = document.createElement('div');
    slot.className = 'slot';
    slot.dataset.index = index;
    slot.addEventListener('dragover', function (e) {
      e.preventDefault();
      e.dataTransfer.dropEffect = dragFrom === null ? 'copy' : 'move';
      slot.classList.add('over');
    });
    slot.addEventListener('dragleave', function () { slot.classList.remove('over'); });
    slot.addEventListener('drop', function (e) {
      e.preventDefault();
      e.stopPropagation();
      slot.classList.remove('over');
      var at = parseInt(slot.dataset.index, 10);
      if (dragFrom !== null) {
        if (at > dragFrom) at -= 1;             // removing shifts later indices
        var moved = steps.splice(dragFrom, 1)[0];
        steps.splice(at, 0, moved);
      } else if (dragOp) {
        steps.splice(at, 0, { op: dragOp.op, params: Object.assign({}, dragOp.params) });
      }
      dragFrom = null; dragOp = null;
      render();
    });
    return slot;
  }

  function makeBlock(step, i) {
    var el = document.createElement('div');
    el.className = 'block';
    el.draggable = true;

    var grip = document.createElement('span');
    grip.className = 'grip';
    grip.textContent = '⠿';
    el.appendChild(grip);

    var name = document.createElement('span');
    name.className = 'bname';
    name.textContent = step.op;
    el.appendChild(name);

    Object.keys(step.params || {}).forEach(function (k) {
      var wrap = document.createElement('label');
      wrap.className = 'bparam';
      wrap.textContent = k;
      var inp = document.createElement('input');
      inp.type = 'number';
      inp.value = step.params[k];
      inp.step = Number.isInteger(step.params[k]) ? 1 : 0.01;
      inp.addEventListener('input', function (e) {
        var v = parseFloat(e.target.value);
        if (!isNaN(v)) { step.params[k] = v; refreshExpr(); }
      });
      // typing in a field must not start a drag
      inp.addEventListener('mousedown', function (e) { e.stopPropagation(); });
      inp.addEventListener('dragstart', function (e) { e.preventDefault(); e.stopPropagation(); });
      wrap.appendChild(inp);
      el.appendChild(wrap);
    });

    var x = document.createElement('button');
    x.className = 'x'; x.type = 'button'; x.textContent = '×';
    x.title = 'remove';
    x.addEventListener('click', function () { steps.splice(i, 1); render(); });
    el.appendChild(x);

    el.addEventListener('dragstart', function (e) {
      dragFrom = i; dragOp = null;
      el.classList.add('dragging');
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', String(i));
    });
    el.addEventListener('dragend', function () {
      el.classList.remove('dragging');
      dragFrom = null;
      Array.prototype.forEach.call(
        canvas.querySelectorAll('.slot'), function (s) { s.classList.remove('over'); });
    });
    return el;
  }

  function render() {
    canvas.innerHTML = '';
    if (!steps.length) {
      var msg = document.createElement('div');
      msg.className = 'dropmsg';
      msg.textContent = 'Drag operators here';
      canvas.appendChild(msg);
      canvas.appendChild(makeSlot(0));
    } else {
      steps.forEach(function (s, i) {
        canvas.appendChild(makeSlot(i));
        canvas.appendChild(makeBlock(s, i));
      });
      canvas.appendChild(makeSlot(steps.length));
    }
    var portfolio = steps.some(function (s) {
      return ['LongShortBalance', 'GrossScale', 'TopN'].indexOf(s.op) >= 0;
    });
    $('blockhint').textContent = steps.length && !portfolio
      ? S.no_portfolio
      : (steps.length + ' ' + S.steps);
    refreshExpr();
  }

  // dropping anywhere on the canvas appends, so a sloppy drop still works
  canvas.addEventListener('dragover', function (e) { e.preventDefault(); });
  canvas.addEventListener('drop', function (e) {
    e.preventDefault();
    if (dragOp) { steps.push({ op: dragOp.op, params: Object.assign({}, dragOp.params) }); }
    dragFrom = null; dragOp = null;
    render();
  });

  /* ---------------- palette ---------------- */

  function paramDefaults(el) {
    var out = {};
    (JSON.parse(el.dataset.params || '[]')).forEach(function (p) { out[p.name] = p.default; });
    return out;
  }

  Array.prototype.forEach.call(document.querySelectorAll('.palette .op'), function (el) {
    el.addEventListener('dragstart', function (e) {
      dragOp = { op: el.dataset.op, params: paramDefaults(el) };
      dragFrom = null;
      e.dataTransfer.effectAllowed = 'copy';
      e.dataTransfer.setData('text/plain', el.dataset.op);
      el.classList.add('dragging');
    });
    el.addEventListener('dragend', function () { el.classList.remove('dragging'); dragOp = null; });
    el.addEventListener('dblclick', function () {
      steps.push({ op: el.dataset.op, params: paramDefaults(el) });
      render();
    });
  });

  $('opsearch').addEventListener('input', function (e) {
    var q = e.target.value.trim().toLowerCase();
    Array.prototype.forEach.call(document.querySelectorAll('.opgroup'), function (g) {
      var any = false;
      Array.prototype.forEach.call(g.querySelectorAll('.op'), function (op) {
        var hit = !q || op.dataset.op.toLowerCase().indexOf(q) >= 0;
        op.hidden = !hit;
        if (hit) any = true;
      });
      g.hidden = !any;
    });
  });

  /* ---------------- tabs ---------------- */

  Array.prototype.forEach.call(document.querySelectorAll('.tab'), function (t) {
    t.addEventListener('click', function () {
      var want = t.dataset.tab;
      if (want === mode) return;
      if (want === 'code') {
        code.value = chainExpression();       // blocks -> code is always possible
        validateCode();
      }
      mode = want;
      Array.prototype.forEach.call(document.querySelectorAll('.tab'), function (b) {
        b.classList.toggle('on', b.dataset.tab === mode);
      });
      $('pane-blocks').hidden = mode !== 'blocks';
      $('pane-code').hidden = mode !== 'code';
      refreshExpr();
    });
  });

  /* ---------------- code editor ---------------- */

  var parseTimer = null;

  function setCodeStatus(cls, text) {
    codestatus.className = 'codestatus ' + cls;
    codestatus.textContent = text;
  }

  async function validateCode() {
    var src = code.value.trim();
    if (!src) { setCodeStatus('', ''); return; }
    var j;
    try {
      var r = await fetch('api/parse', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ expression: src, lang: CFG.lang })
      });
      j = await r.json();
    } catch (err) { setCodeStatus('bad', String(err)); return; }

    if (!j.ok) { setCodeStatus('bad', j.error); return; }
    if (j.chain) {
      steps = j.chain.steps.map(function (s) { return { op: s.op, params: s.params || {} }; });
      var sel = $('source');
      var want = j.chain.source === 'px' ? 'close' : j.chain.source;
      if (Array.prototype.some.call(sel.options, function (o) { return o.value === want; })) {
        sel.value = want;
      }
      render();                      // the canvas must show what the code says
      setCodeStatus('good', S.valid_blocks);
    } else {
      setCodeStatus('good', S.valid_tree);
    }
    refreshExpr();
  }

  code.addEventListener('input', function () {
    clearTimeout(parseTimer);
    parseTimer = setTimeout(validateCode, 300);
    refreshExpr();
  });
  code.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); run(); }
    if (e.key === 'Tab') {                       // Tab indents, does not escape
      e.preventDefault();
      var s = code.selectionStart;
      code.value = code.value.slice(0, s) + '  ' + code.value.slice(code.selectionEnd);
      code.selectionStart = code.selectionEnd = s + 2;
    }
  });

  /* ---------------- presets & run ---------------- */

  Array.prototype.forEach.call(document.querySelectorAll('.preset'), function (b) {
    b.addEventListener('click', function () {
      var p = PRESETS[b.dataset.preset];
      $('source').value = p.source;
      steps = JSON.parse(JSON.stringify(p.steps)).map(function (s) {
        return { op: s.op, params: s.params || {} };
      });
      if (mode === 'code') { code.value = chainExpression(); validateCode(); }
      render();
      run();
    });
  });

  $('clear').addEventListener('click', function () {
    steps = []; render(); results.hidden = true; setStatus('', false);
  });
  $('source').addEventListener('change', refreshExpr);
  $('run').addEventListener('click', run);

  var COST_IDS = ['brokerage', 'tax_sell', 'half_spread', 'borrow_annual'];

  function costRates() {
    // the form is in bps (and %/yr for borrow); the API wants fractions
    var n = function (id, div) {
      var v = parseFloat($(id).value);
      return (isFinite(v) && v >= 0) ? v / div : CFG.defaults[id] / div;
    };
    return {
      brokerage: n('brokerage', 1e4),
      tax_sell: n('tax_sell', 1e4),
      half_spread: n('half_spread', 1e4),
      borrow_annual: n('borrow_annual', 100)
    };
  }

  function refreshCost() {
    var r = costRates();
    var tax = $('day_trade').checked ? r.tax_sell / 2 : r.tax_sell;
    var rt = (2 * r.brokerage + tax + 2 * r.half_spread) * 1e4;
    $('rt').textContent = rt.toFixed(1) + ' bps';
  }

  COST_IDS.forEach(function (id) { $(id).addEventListener('input', refreshCost); });
  $('day_trade').addEventListener('change', refreshCost);
  $('resetcosts').addEventListener('click', function () {
    COST_IDS.forEach(function (id) {
      $(id).value = CFG.defaults[id].toFixed(id === 'borrow_annual' ? 2 : 1);
    });
    refreshCost();
  });

  function setStatus(msg, bad) {
    status.hidden = !msg;
    status.textContent = msg || '';
    status.className = 'status' + (bad ? ' bad' : '');
  }

  async function run() {
    var rates = costRates();
    var body = {
      lang: CFG.lang,
      options: {
        exec_shift: parseInt($('exec_shift').value, 10),
        brokerage: rates.brokerage,
        tax_sell: rates.tax_sell,
        half_spread: rates.half_spread,
        borrow_annual: rates.borrow_annual,
        tradability: $('tradability').checked,
        day_trade: $('day_trade').checked
      }
    };
    if (mode === 'code') {
      if (!code.value.trim()) { setStatus(S.write_expr, true); return; }
      body.expression = code.value.trim();
    } else {
      if (!steps.length) { setStatus(S.add_operator, true); return; }
      body.source = $('source').value;
      body.steps = steps;
    }

    setStatus(S.running, false);
    $('run').disabled = true;
    var r, j;
    try {
      r = await fetch('api/run', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      j = await r.json();
    } catch (err) {
      setStatus('Request failed: ' + err, true);
      $('run').disabled = false;
      return;
    }
    $('run').disabled = false;

    if (!r.ok) {
      setStatus(j.error || 'failed', true);
      if (mode === 'code' && j.pos != null) {
        code.focus();
        code.setSelectionRange(j.pos, Math.min(j.pos + 1, code.value.length));
      }
      results.hidden = true;
      return;
    }

    setStatus(S.done_in + ' ' + j.elapsed_ms + ' ms', false);
    expr.textContent = 'sig = ' + j.expression;

    var v = $('verdict'); v.innerHTML = '';
    j.verdict.forEach(function (item) {
      var li = document.createElement('li');
      li.className = item.fail ? 'fail' : '';
      li.textContent = item.text;
      v.appendChild(li);
    });

    Object.keys(j.charts).forEach(function (k) {
      var node = $('c_' + k);
      if (node) node.innerHTML = j.charts[k];
    });
    // a panel with no target element would silently vanish, so say so
    Object.keys(j.charts).forEach(function (k) {
      if (!$('c_' + k)) console.warn('no slot for chart', k);
    });

    var st = $('stats'); st.innerHTML = '';
    j.stats.forEach(function (row) {
      var tr = document.createElement('tr');
      var a = document.createElement('td'); a.textContent = row[0];
      var b = document.createElement('td'); b.textContent = row[1];
      tr.appendChild(a); tr.appendChild(b); st.appendChild(tr);
    });

    var yt = $('yeartbl'); yt.innerHTML = '';
    j.by_year.forEach(function (row) {
      var tr = document.createElement('tr');
      row.forEach(function (c) {
        var td = document.createElement('td'); td.textContent = c; tr.appendChild(td);
      });
      yt.appendChild(tr);
    });

    results.hidden = false;
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  /* First paint shows a worked example rather than an empty canvas. It is a
     real factor with a real verdict attached -- including, at these costs, the
     fact that it loses money -- which is a more honest landing page than a
     blank form or a curated winner. */
  function boot() {
    refreshCost();
    render();
    if (CFG.example && CFG.example.expression) {
      code.value = CFG.example.expression;
      validateCode().then(function () { run(); });
    }
  }

  boot();
})();
