(function () {
  'use strict';

  var DATA_URL = './data/examples.json';
  var examples = [];
  var baselines = [];
  var judgeTypes = null;
  var judgeCriteria = null;
  var current = -1;

  // ---- DOM helpers (all dataset text goes through textContent) ----
  function el(tag, attrs) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === 'class') node.className = attrs[k];
        else if (k === 'text') node.textContent = attrs[k];
        else node.setAttribute(k, attrs[k]);
      });
    }
    for (var i = 2; i < arguments.length; i++) {
      var c = arguments[i];
      if (c === null || c === undefined || c === false) continue;
      node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    }
    return node;
  }

  function link(href, text) {
    return el('a', { href: href, target: '_blank', rel: 'noopener', text: text });
  }

  function isBlank(v) {
    return v === null || v === undefined || (typeof v === 'string' && v.trim() === '') ||
      (Array.isArray(v) && v.length === 0);
  }

  function show(v) {
    if (Array.isArray(v)) v = v.join(', ');
    return isBlank(v) ? '—' : String(v);
  }

  // Touch screens have no hover; popovers open on tap there.
  var POINT_VERB = window.matchMedia && window.matchMedia('(hover: none)').matches ? 'Tap' : 'Hover';

  function icon(name) { return el('span', { class: 'icon is-small v-icon' }, el('i', { class: 'fas fa-' + name })); }

  // ---- Popovers: details open on hover, keyboard focus, or tap ----
  var tipContent = new WeakMap();
  var pop = null;
  var owner = null;
  var hideTimer = null;

  // content is a function returning the popover's DOM node; it runs each time the popover opens.
  function withTip(node, content) {
    node.classList.add('has-tip');
    if (!node.hasAttribute('tabindex')) node.setAttribute('tabindex', '0');
    tipContent.set(node, content);
    return node;
  }

  function help(content, ariaLabel) {
    return withTip(el('span', { class: 'v-help', role: 'button', 'aria-label': ariaLabel || 'More information',
      text: '?' }), content);
  }

  function popTitle(text) { return el('p', { class: 'v-pop-title', text: text }); }

  // Definition list; pairs with blank values are left out.
  function dl(pairs) {
    var d = el('dl', { class: 'v-pop-dl' });
    pairs.forEach(function (p) {
      if (isBlank(p[1])) return;
      d.appendChild(el('dt', { text: p[0] }));
      d.appendChild(el('dd', { text: show(p[1]) }));
    });
    return d;
  }

  function placeTip() {
    var r = owner.getBoundingClientRect();
    var vw = document.documentElement.clientWidth;
    var vh = window.innerHeight;
    pop.style.left = '0px';
    pop.style.top = '0px';
    var w = pop.offsetWidth;
    var h = pop.offsetHeight;
    var left = Math.max(8, Math.min(r.left + r.width / 2 - w / 2, vw - w - 8));
    var top = r.bottom + 8;
    if (top + h > vh - 8 && r.top - 8 - h >= 8) top = r.top - 8 - h;
    pop.style.left = (left + window.scrollX) + 'px';
    pop.style.top = (top + window.scrollY) + 'px';
  }

  function openTip(t) {
    clearTimeout(hideTimer);
    if (owner === t) return;
    closeTip();
    var content = tipContent.get(t);
    if (!content) throw new Error('popover trigger has no content');
    owner = t;
    pop.textContent = '';
    pop.appendChild(content());
    pop.hidden = false;
    t.classList.add('is-tip-open');
    t.setAttribute('aria-describedby', pop.id);
    placeTip();
  }

  function closeTip() {
    clearTimeout(hideTimer);
    if (!owner) return;
    owner.classList.remove('is-tip-open');
    owner.removeAttribute('aria-describedby');
    owner = null;
    pop.hidden = true;
  }

  function initTips() {
    pop = el('div', { id: 'v-pop', class: 'v-pop', role: 'tooltip' });
    pop.hidden = true;
    document.body.appendChild(pop);
    // Static triggers in index.html take their content from a <template>.
    document.querySelectorAll('[data-tip-template]').forEach(function (node) {
      var tpl = document.getElementById(node.getAttribute('data-tip-template'));
      if (!tpl) throw new Error('missing popover template ' + node.getAttribute('data-tip-template'));
      withTip(node, function () { return tpl.content.cloneNode(true); });
    });

    function trigger(e) { return e.target.closest ? e.target.closest('.has-tip') : null; }
    document.addEventListener('mouseover', function (e) {
      var t = trigger(e);
      if (t) openTip(t);
      else if (pop.contains(e.target)) clearTimeout(hideTimer);
    });
    document.addEventListener('mouseout', function (e) {
      if (!owner) return;
      var to = e.relatedTarget;
      if (to && (owner.contains(to) || pop.contains(to))) return;
      if (owner.contains(e.target) || pop.contains(e.target)) {
        clearTimeout(hideTimer);
        hideTimer = setTimeout(closeTip, 150);
      }
    });
    document.addEventListener('focusin', function (e) {
      var t = trigger(e);
      if (t) openTip(t);
      else if (!pop.contains(e.target)) closeTip();
    });
    document.addEventListener('click', function (e) {
      var t = trigger(e);
      if (t) openTip(t);
      else if (!pop.contains(e.target)) closeTip();
    });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeTip(); });
    window.addEventListener('resize', closeTip);
  }

  // ---- Rendering ----
  function renderChips() {
    var chips = document.getElementById('chips');
    chips.textContent = '';
    examples.forEach(function (ex, i) {
      var a = el('a', { class: 'chip', href: '#' + ex.id, title: ex.id },
        el('span', { class: 'chip-num', text: String(i + 1) }), ex.aspect);
      chips.appendChild(a);
    });
  }

  function creditLine(ex) {
    var src = ex.source;
    var a = ex.attribution;
    var p = el('p', { class: 'v-attribution' });
    function text(t) { p.appendChild(document.createTextNode(t)); }
    if (a && !isBlank(a.title)) {
      text('“');
      p.appendChild(isBlank(a.url) ? document.createTextNode(a.title) : link(a.url, a.title));
      text('”');
      if (!isBlank(a.authors)) text(' by ' + show(a.authors));
      if (!isBlank(a.illustrators)) text(', illustrated by ' + show(a.illustrators));
      text(' · ');
      p.appendChild(link(src.url, src.name));
      text(' · ' + (isBlank(a.license) ? src.license + ' (collection license)' : a.license));
    } else {
      text('Page images and narration from ');
      p.appendChild(link(src.url, src.name));
      text(' · ' + src.license);
    }
    p.appendChild(help(function () {
      var licenseNote = a && !isBlank(a.license) ? 'License: ' + a.license + '.'
        : src.name + '’s collection license (' + src.license + '); not verified for this specific book.';
      return el('div', null,
        popTitle('Source'),
        el('p', { text: licenseNote }),
        el('p', { text: 'Page images resized for the web.' }),
        el('p', null, 'Sample ID: ', el('code', { text: ex.id })));
    }, 'Source and license details'));
    return p;
  }

  function pageFigure(page, caption) {
    return el('figure', { class: 'v-figure' },
      el('img', { src: './data/' + page.image, alt: caption }),
      el('figcaption', { text: caption }));
  }

  function label(text, helpContent) {
    return el('p', { class: 'v-label' }, text, helpContent ? help(helpContent) : null);
  }

  function colTitle(text, cls, helpText) {
    return el('p', { class: 'v-col-title ' + cls }, text,
      help(function () { return el('p', { text: helpText }); }));
  }

  function tag(text, iconName, content) {
    return withTip(el('span', { class: 'v-tag' }, iconName ? icon(iconName) : null, text), content);
  }

  function conditionTags(cond) {
    var tags = el('div', { class: 'v-tags' });
    cond.characters.forEach(function (c) {
      tags.appendChild(tag(c.name, 'user', function () {
        return el('div', null, popTitle(c.name),
          dl([['Emotion', c.emotion], ['Visibility', c.visibility], ['Action', c.action],
            ['Speech intent', c.speech_intent]]));
      }));
    });
    if (!isBlank(cond.scene)) {
      tags.appendChild(tag('Scene', 'image', function () {
        return el('div', null, popTitle('Scene'), el('p', { text: cond.scene }));
      }));
    }
    if (!isBlank(cond.ambient_sound)) {
      tags.appendChild(tag('Ambient sound', 'volume-up', function () {
        return el('div', null, popTitle('Ambient sound'), el('p', { text: cond.ambient_sound }));
      }));
    }
    return tags;
  }

  function bookMetadata(meta) {
    var summary = [meta.genre, meta.narrative_perspective, isBlank(meta.narrative_tense) ? null : meta.narrative_tense + ' tense']
      .filter(function (v) { return !isBlank(v); }).join(' · ');
    return el('p', { class: 'v-bookline' },
      el('span', { class: 'v-bookline-label', text: 'Book: ' }), summary,
      help(function () {
        var chars = el('div', { class: 'v-pop-chars' });
        meta.characters.forEach(function (c) {
          var facts = [c.role, c.species, c.sex, c.age_range].filter(function (v) { return !isBlank(v); }).join(', ');
          chars.appendChild(el('p', null, el('b', { text: c.name }), facts ? ' — ' + facts : '',
            isBlank(c.personality) ? null : el('br'), isBlank(c.personality) ? null : 'Personality: ' + show(c.personality),
            isBlank(c.appearance) ? null : el('br'), isBlank(c.appearance) ? null : 'Appearance: ' + show(c.appearance)));
        });
        return el('div', null, popTitle('Book-level metadata'),
          dl([['Genre', meta.genre], ['Topic', meta.topic], ['Style', meta.style],
            ['Tense', meta.narrative_tense], ['Perspective', meta.narrative_perspective]]),
          meta.characters.length ? popTitle('Characters') : null, chars);
      }, 'Book-level metadata'));
  }

  function speechLine(speech) {
    var parts = [speech.speaker, speech.emotion,
      isBlank(speech.speed) ? null : speech.speed + ' speed',
      isBlank(speech.pitch) ? null : speech.pitch + ' pitch', speech.gender];
    return el('p', { class: 'v-subline', text: parts.filter(function (v) { return !isBlank(v); }).join(' · ') });
  }

  // ---- Model outputs ----
  var MODALITY_NOUN = { text: 'narration', image: 'image', speech: 'speech' };

  function missingBox(modality, result) {
    var title = result.status === 'empty' ? 'Empty ' + MODALITY_NOUN[modality]
      : 'No ' + MODALITY_NOUN[modality] + ' output';
    return el('div', { class: 'v-missing v-missing-' + modality },
      el('span', { class: 'v-missing-title' }, title,
        result.reason ? help(function () { return el('p', { text: result.reason }); }, 'Reason') : null));
  }

  function outputPart(modality, result, alt) {
    if (!result) throw new Error('missing ' + modality + ' entry in model outputs');
    if (result.status !== 'ok') return missingBox(modality, result);
    if (modality === 'image') {
      return el('figure', { class: 'v-figure v-out-image' }, el('img', { src: './data/' + result.path, alt: alt }));
    }
    if (modality === 'text') return el('blockquote', { class: 'v-quote v-out-text', text: result.text });
    return el('audio', { class: 'v-out-audio', controls: '', preload: 'metadata', src: './data/' + result.path });
  }

  function expertsLine(b) {
    var parts = [];
    if (b.image_expert) parts.push(b.image_expert + ' (image)');
    if (b.speech_expert) parts.push(b.speech_expert + ' (speech)');
    return parts.length ? '+ ' + parts.join(' + ') : 'Single model, no external experts';
  }

  // ---- Judge scores ----
  function scorePopover(jtype, summary) {
    var labels = judgeCriteria[jtype];
    var n = summary.judges.length;
    var head = el('tr');
    labels.forEach(function (l) { head.appendChild(el('th', { class: 'is-num', text: l })); });
    head.appendChild(el('th', { class: 'is-num', text: 'Mean' }));
    var body = el('tbody');
    summary.judges.forEach(function (j) {
      // Judge name on its own row so the score columns fit narrow screens.
      body.appendChild(el('tr', { class: 'v-judge-row' },
        el('td', { class: 'v-judge-name', colspan: String(labels.length + 1), text: j.model })));
      var tr = el('tr', { class: 'v-judge-scores' });
      if (j.status === 'ok') {
        j.scores.forEach(function (sc) { tr.appendChild(el('td', { class: 'is-num', text: String(sc) })); });
        tr.appendChild(el('td', { class: 'is-num v-judge-mean', text: j.mean.toFixed(2) }));
      } else {
        tr.appendChild(el('td', { class: 'v-judge-status', colspan: String(labels.length + 1),
          text: j.status === 'skipped' ? 'not scored (output missing)' : 'response could not be parsed' }));
      }
      body.appendChild(tr);
    });
    var note;
    if (summary.n_valid === 0) note = 'Not scored by any judge.';
    else if (summary.n_valid < n) note = 'Mean of the criteria, averaged over the ' + summary.n_valid + ' of ' + n +
      ' judges whose responses could be scored.';
    else note = 'Mean of the criteria, averaged over the ' + n + ' judges.';
    return el('div', null,
      popTitle(judgeTypes[jtype] + ' · ' + (summary.n_valid ? summary.mean.toFixed(1) : 'not scored')),
      jtype === 'joint' ? el('p', { text: 'Judges the image, narration, and speech together.' }) : null,
      el('p', { class: 'v-pop-note', text: note + ' Scale 1–10.' }),
      el('table', { class: 'table is-narrow v-score-table' }, el('thead', null, head), body));
  }

  function scoreBadge(jtype, summary) {
    if (!summary) throw new Error('missing judge summary');
    var badge;
    if (summary.n_valid === 0) {
      badge = el('span', { class: 'v-score is-none', text: '—' });
    } else {
      badge = el('span', { class: 'v-score',
        text: summary.mean.toFixed(1) + (summary.n_valid < summary.judges.length ? '*' : '') });
    }
    badge.setAttribute('aria-label', judgeTypes[jtype] + ' judge score details');
    return withTip(badge, function () { return scorePopover(jtype, summary); });
  }

  function partHead(jtype, summary) {
    return el('div', { class: 'v-part-head' },
      el('span', { class: 'v-label', text: judgeTypes[jtype] }), scoreBadge(jtype, summary));
  }

  function outputsHelp() {
    return el('div', null,
      el('p', null, 'Each card is the highest-scoring configuration of its paradigm by Total Average (',
        el('a', { href: '../#results', text: 'full results' }), ').'),
      el('p', { text: 'Scores are this sample’s LLM-judge ratings (1–10): the mean of four rubric criteria, ' +
        'averaged over three judges. Integrated judges all three outputs together.' }),
      el('p', { text: 'The dataset has no reference audio, so speech is judged against the ground-truth utterance ' +
        'and its speech metadata.' }));
  }

  function renderOutputs(ex) {
    if (!ex.outputs) throw new Error(ex.id + ' has no model outputs');
    var grid = el('div', { class: 'columns is-multiline' });
    baselines.forEach(function (b) {
      var out = ex.outputs[b.key];
      if (!out) throw new Error(ex.id + ' has no outputs for ' + b.key);
      if (!out.judges) throw new Error(ex.id + ' has no judge scores for ' + b.key);
      grid.appendChild(el('div', { class: 'column is-one-third-desktop is-half-tablet' },
        el('div', { class: 'v-card' },
          el('div', { class: 'v-card-head' },
            el('span', { class: 'paradigm-tag ' + b.paradigm, text: b.paradigm_label }),
            el('p', { class: 'v-card-title', text: b.backbone }),
            el('p', { class: 'v-card-sub', text: expertsLine(b) })),
          partHead('image', out.judges.image),
          outputPart('image', out.image, b.backbone + ' generated image'),
          partHead('text', out.judges.text),
          outputPart('text', out.text),
          partHead('speech', out.judges.speech),
          outputPart('speech', out.speech),
          el('div', { class: 'v-integrated' }, partHead('joint', out.judges.joint)))));
    });
    return el('div', { class: 'v-outputs' },
      el('div', { class: 'v-outputs-head' },
        el('h3', { class: 'title is-5' }, 'Model outputs', help(outputsHelp, 'About the model outputs')),
        el('span', { class: 'v-outputs-hint', text: POINT_VERB + ' a score for each judge’s ratings.' })),
      grid);
  }

  function renderExample(i) {
    closeTip();
    var ex = examples[i];
    current = i;
    var root = document.getElementById('example');
    root.textContent = '';

    var cond = ex.next_page_condition;
    var speech = ex.ground_truth.speech;

    root.appendChild(el('div', { class: 'v-header' },
      el('p', { class: 'v-counter', text: 'Example ' + (i + 1) + ' of ' + examples.length }),
      el('h2', { class: 'title is-4' }, ex.aspect,
        help(function () { return el('div', null, popTitle('Why this example'), el('p', { text: ex.rationale })); },
          'Why this example')),
      creditLine(ex)));

    var input = el('div', { class: 'column is-half' },
      colTitle('Input', 'is-input', 'What every model receives: the current page (illustration and narration), ' +
        'book-level metadata, and the condition for the next page.'),
      pageFigure(ex.input.current_page, 'Current page (' + ex.transition.from + ')'),
      label('Narration'),
      el('blockquote', { class: 'v-quote', text: ex.input.current_page.text }),
      label('Next-page condition', function () {
        return el('p', { text: 'What the next page should contain: the narration instruction below, plus the ' +
          'characters, scene, and ambient sound. ' + POINT_VERB + ' a tag for its details.' });
      }),
      el('p', { class: 'v-instruction', text: cond.text_instruction }),
      conditionTags(cond),
      bookMetadata(ex.input.book_metadata));

    var gt = el('div', { class: 'column is-half' },
      colTitle('Ground truth', 'is-gt', 'The book’s actual next page, plus the annotated character ' +
        'utterance and its speech metadata. The dataset has no reference audio.'),
      pageFigure(ex.ground_truth.next_page, 'Next page (' + ex.transition.to + ')'),
      label('Narration'),
      el('blockquote', { class: 'v-quote', text: ex.ground_truth.next_page.text }),
      label('Speech'),
      el('blockquote', { class: 'v-quote is-speech', text: '“' + speech.utterance + '”' }),
      speechLine(speech));

    root.appendChild(el('div', { class: 'columns v-columns' }, input, gt));
    root.appendChild(renderOutputs(ex));

    Array.prototype.forEach.call(document.querySelectorAll('#chips .chip'), function (c, j) {
      c.classList.toggle('is-active', j === i);
    });
    document.title = 'Omni-StoryBench Viewer – ' + ex.aspect;
  }

  function showError(message) {
    var root = document.getElementById('example');
    root.textContent = '';
    root.appendChild(el('div', { class: 'notification is-danger is-light' }, el('strong', { text: 'Error: ' }), message));
  }

  function route() {
    var id = decodeURIComponent(location.hash.slice(1));
    if (!id) {
      history.replaceState(null, '', '#' + examples[0].id);
      return renderExample(0);
    }
    for (var i = 0; i < examples.length; i++) {
      if (examples[i].id === id) return renderExample(i);
    }
    current = -1;
    closeTip();
    showError('Example “' + id + '” is not in this viewer. Pick one of the examples above.');
  }

  function go(delta) {
    var n = examples.length;
    var i = current < 0 ? 0 : (current + delta + n) % n;
    location.hash = '#' + examples[i].id;
  }

  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.navbar-burger').forEach(function (burger) {
      burger.addEventListener('click', function () {
        burger.classList.toggle('is-active');
        document.querySelectorAll('.navbar-menu').forEach(function (m) { m.classList.toggle('is-active'); });
      });
    });
    initTips();

    fetch(DATA_URL)
      .then(function (r) {
        if (!r.ok) throw new Error('could not load ' + DATA_URL + ' (HTTP ' + r.status + ')');
        return r.json();
      })
      .then(function (data) {
        if (!data.examples || !data.examples.length) throw new Error(DATA_URL + ' has no examples');
        examples = data.examples;
        if (!Array.isArray(data.baselines)) throw new Error(DATA_URL + ' has no baselines list');
        baselines = data.baselines;
        if (!data.judge_types || !data.judge_criteria) throw new Error(DATA_URL + ' has no judge metadata');
        judgeTypes = data.judge_types;
        judgeCriteria = data.judge_criteria;
        renderChips();
        document.getElementById('prev').addEventListener('click', function () { go(-1); });
        document.getElementById('next').addEventListener('click', function () { go(1); });
        document.addEventListener('keydown', function (e) {
          if (e.target.closest && e.target.closest('input, textarea, select')) return;
          if (e.key === 'ArrowLeft') go(-1);
          else if (e.key === 'ArrowRight') go(1);
        });
        window.addEventListener('hashchange', route);
        route();
      })
      .catch(function (err) {
        showError(err.message);
        throw err;
      });
  });
})();
