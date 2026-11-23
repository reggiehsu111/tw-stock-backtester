/* Futures tearsheet: same shape as the builder, one instrument. */
(function () {
  'use strict';
  var CFG = JSON.parse(document.getElementById('cfg').textContent);
  var S = CFG.strings;
  function $(id) { return document.getElementById(id); }
  var status = $('status'), results = $('results');

  function setStatus(msg, bad) {
    status.hidden = !msg;
    status.textContent = msg || '';
    status.className = 'status' + (bad ? ' bad' : '');
  }

  async function run() {
    setStatus(S.running, false);
    $('run').disabled = true;
    var body = {
      lang: CFG.lang,
      strategy: $('strategy').value,
      normalization: $('normalization').value,
      contract: $('contract').value,
      exec_shift: parseInt($('exec_shift').value, 10),
      entry_z: $('entry_z').value,
      lookback: $('lookback').value,
      hold: $('hold').value,
      spread_bps: $('spread_bps').value,
      event_set: $('event_set').value
    };
    var r, j;
    try {
      r = await fetch('api/futures/run', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      j = await r.json();
    } catch (err) { setStatus('Request failed: ' + err, true); $('run').disabled = false; return; }
    $('run').disabled = false;
    if (!r.ok) { setStatus(j.error || 'failed', true); results.hidden = true; return; }

    setStatus(S.done_in + ' ' + j.elapsed_ms + ' ms', false);
    $('desc').textContent = j.describe + '\n' + j.costs;

    var v = $('verdict'); v.innerHTML = '';
    j.verdict.forEach(function (it) {
      var li = document.createElement('li');
      li.className = it.fail ? 'fail' : '';
      li.textContent = it.text; v.appendChild(li);
    });

    Object.keys(j.charts).forEach(function (k) {
      var n = $('c_' + k); if (n) n.innerHTML = j.charts[k];
    });

    function fill(id, rows) {
      var t = $(id); t.innerHTML = '';
      rows.forEach(function (row) {
        var tr = document.createElement('tr');
        row.forEach(function (c) {
          var td = document.createElement('td'); td.textContent = c; tr.appendChild(td);
        });
        t.appendChild(tr);
      });
    }
    fill('stats', j.stats);
    fill('consist', j.consistency);
    fill('wf', j.walk_forward);
    results.hidden = false;
  }

  $('run').addEventListener('click', run);
  ['strategy', 'normalization', 'contract', 'event_set'].forEach(function (id) {
    $(id).addEventListener('change', run);
  });
  run();
})();
