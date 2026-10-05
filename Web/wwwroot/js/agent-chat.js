(function () {
    'use strict';

    var root = document.getElementById('agentChat');
    if (!root) { return; }

    var STORE_KEY = 'cyclonAgentChat';
    var TAB_KEY = 'cyclonAgentTab';
    var OPEN_KEY = 'cyclonAgentOpen';
    var MAX_ENTRIES = 50;
    var QUEUE_NOTE_DELAY_MS = 4000;

    var el = {
        launcher: document.getElementById('agentLauncher'),
        panel: document.getElementById('agentPanel'),
        close: document.getElementById('agentClose'),
        reset: document.getElementById('agentReset'),
        messages: document.getElementById('agentMessages'),
        chips: document.getElementById('agentChips'),
        input: document.getElementById('agentInput'),
        send: document.getElementById('agentSend'),
        draftWrap: document.getElementById('agentDraftWrap'),
        draftToggle: document.getElementById('agentDraftToggle'),
        draftBody: document.getElementById('agentDraftBody'),
        draftCount: document.getElementById('agentDraftCount'),
        save: document.getElementById('agentSave'),
        saveProject: document.getElementById('agentProject'),
        saveTag: document.getElementById('agentTag'),
        saveName: document.getElementById('agentName'),
        saveError: document.getElementById('agentSaveError'),
        saveCancel: document.getElementById('agentSaveCancel'),
        saveConfirm: document.getElementById('agentSaveConfirm')
    };

    var urls = {
        message: root.getAttribute('data-message-url'),
        reset: root.getAttribute('data-reset-url'),
        projects: root.getAttribute('data-projects-url'),
        save: root.getAttribute('data-save-url')
    };

    var WELCOME = 'Hi! Tell me about your gas flow, particle size and density, and the cyclone type you have in mind, and I will size the cyclone for you.';

    var state = { entries: [], draft: [], busy: false, tabId: getTabId() };

    /* ── Storage ──────────────────────────────────────────────────────── */

    function getTabId() {
        try {
            var id = sessionStorage.getItem(TAB_KEY);
            if (!id) {
                id = 't' + Math.random().toString(36).slice(2, 12) + Date.now().toString(36);
                sessionStorage.setItem(TAB_KEY, id);
            }
            return id;
        } catch (e) {
            return 'default';
        }
    }

    function persist() {
        try {
            sessionStorage.setItem(STORE_KEY, JSON.stringify({
                entries: state.entries.slice(-MAX_ENTRIES),
                draft: state.draft
            }));
        } catch (e) { /* storage unavailable */ }
    }

    function restore() {
        try {
            var raw = sessionStorage.getItem(STORE_KEY);
            if (!raw) { return; }
            var data = JSON.parse(raw);
            if (Array.isArray(data.entries)) { state.entries = data.entries; }
            if (Array.isArray(data.draft)) { state.draft = data.draft; }
        } catch (e) { /* ignore corrupt state */ }
    }

    /* ── Helpers ──────────────────────────────────────────────────────── */

    function node(tag, className, text) {
        var n = document.createElement(tag);
        if (className) { n.className = className; }
        if (text !== undefined && text !== null) { n.textContent = text; }
        return n;
    }

    function fmt(value, digits) {
        var n = Number(value);
        if (!isFinite(n)) { return '—'; }
        return n.toLocaleString(undefined, { maximumFractionDigits: digits === undefined ? 1 : digits });
    }

    function antiForgeryToken() {
        var input = root.querySelector('input[name="__RequestVerificationToken"]');
        return input ? input.value : '';
    }

    function scrollToBottom() {
        el.messages.scrollTop = el.messages.scrollHeight;
    }

    function request(method, url, body) {
        var headers = { 'Accept': 'application/json' };
        var options = { method: method, headers: headers, credentials: 'same-origin' };

        if (body !== undefined) {
            headers['Content-Type'] = 'application/json';
            headers['RequestVerificationToken'] = antiForgeryToken();
            options.body = JSON.stringify(body);
        }

        return fetch(url, options).then(function (res) {
            return res.text().then(function (text) {
                var data = null;
                try { data = text ? JSON.parse(text) : null; } catch (e) { data = null; }

                if (!res.ok) {
                    var message = (data && data.error) || 'Something went wrong (' + res.status + ').';
                    if (res.status === 401 || res.status === 403) {
                        message = 'Your session has expired or you do not have access. Please sign in again.';
                    }
                    var err = new Error(message);
                    err.status = res.status;
                    throw err;
                }
                return data;
            });
        });
    }

    /* ── Panel ────────────────────────────────────────────────────────── */

    function openPanel() {
        root.classList.add('is-open');
        el.panel.setAttribute('aria-hidden', 'false');
        try { sessionStorage.setItem(OPEN_KEY, '1'); } catch (e) { }
        scrollToBottom();
        el.input.focus();
    }

    function closePanel() {
        root.classList.remove('is-open');
        el.panel.setAttribute('aria-hidden', 'true');
        try { sessionStorage.removeItem(OPEN_KEY); } catch (e) { }
        el.launcher.focus();
    }

    /* ── Draft progress ───────────────────────────────────────────────── */

    function renderDraft() {
        var fields = state.draft || [];
        var anyValue = fields.some(function (f) { return f.value; });

        el.draftWrap.hidden = !anyValue;
        if (!anyValue) { return; }

        var mandatory = fields.filter(function (f) { return f.mandatory; });
        var filled = mandatory.filter(function (f) { return f.value; }).length;
        el.draftCount.textContent = filled + ' / ' + mandatory.length;

        el.draftBody.textContent = '';
        fields.forEach(function (f) {
            if (!f.mandatory && !f.value) { return; }

            var cell = node('div', 'agent-field ' + (f.value ? 'is-filled' : 'is-missing'));
            cell.appendChild(node('span', 'agent-field-label', f.label));
            var value = node('span', 'agent-field-value', f.value || 'needed');
            value.title = f.value || 'needed';
            cell.appendChild(value);
            el.draftBody.appendChild(cell);
        });
    }

    /* ── Message rendering ────────────────────────────────────────────── */

    function renderBubble(kind, text) {
        var cls = kind === 'user' ? 'agent-msg agent-msg-user'
            : kind === 'error' ? 'agent-msg agent-msg-bot agent-msg-error'
                : 'agent-msg agent-msg-bot';
        var row = node('div', cls);
        row.appendChild(node('div', 'agent-bubble', text));
        return row;
    }

    function renderLink(entry) {
        var row = node('div', 'agent-msg agent-msg-bot');
        var bubble = node('div', 'agent-bubble');
        bubble.appendChild(document.createTextNode(entry.text + ' '));

        if (typeof entry.url === 'string' && entry.url.charAt(0) === '/' && entry.url.charAt(1) !== '/') {
            var a = node('a', null, 'Open results');
            a.href = entry.url;
            bubble.appendChild(a);
        }

        row.appendChild(bubble);
        return row;
    }

    function kpi(label, value, unit) {
        var box = node('div', 'agent-kpi');
        box.appendChild(node('div', 'agent-kpi-label', label));
        var v = node('div', 'agent-kpi-value', value);
        if (unit) { v.appendChild(node('span', 'agent-kpi-unit', unit)); }
        box.appendChild(v);
        return box;
    }

    function verdictOf(checks) {
        var list = Array.isArray(checks) ? checks : [];
        var hasFail = list.some(function (c) { return String(c.status).toUpperCase() === 'FAIL'; });
        var hasWarn = list.some(function (c) { return String(c.status).toUpperCase() === 'WARN'; });

        if (hasFail) { return { text: 'Verdict: NOT ACCEPTABLE', cls: 'bad' }; }
        if (hasWarn) { return { text: 'Verdict: ACCEPTABLE WITH CAUTION', cls: 'caution' }; }
        return { text: 'Verdict: ACCEPTABLE', cls: 'ok' };
    }

    function renderResult(entry, showActions) {
        var r = entry.result || {};
        var checks = Array.isArray(entry.checks) ? entry.checks : [];

        var row = node('div', 'agent-msg agent-msg-bot');
        var card = node('div', 'agent-result');

        var head = node('div', 'agent-result-head');
        head.appendChild(node('span', 'agent-result-title', 'Design result'));
        head.appendChild(node('span', 'agent-type-badge', r.cycloneTypeName || r.cycloneType || 'Cyclone'));
        card.appendChild(head);

        var grid = node('div', 'agent-kpis');
        grid.appendChild(kpi('Efficiency', fmt(r.efficiencyPercent, 1), '%'));
        grid.appendChild(kpi('Pressure drop', fmt(r.pressureDropPa, 0), 'Pa'));
        grid.appendChild(kpi('Cut diameter', fmt(r.cutDiameterMicron, 1), 'µm'));
        grid.appendChild(kpi('Inlet velocity', fmt(r.inletVelocityMs, 1), 'm/s'));
        grid.appendChild(kpi('Flow', fmt(r.flowRateM3hr, 0), 'm³/h'));
        grid.appendChild(kpi('Barrel Ø', fmt(r.barrelDiameterMm, 0), 'mm'));
        card.appendChild(grid);

        var dims = node('div', 'agent-dims');
        dims.appendChild(document.createTextNode('Height '));
        dims.appendChild(node('span', 'agent-mono', fmt(r.totalHeightMm, 0) + ' mm'));
        dims.appendChild(document.createTextNode(' · Inlet '));
        dims.appendChild(node('span', 'agent-mono', fmt(r.inletHeightMm, 0) + ' × ' + fmt(r.inletWidthMm, 0) + ' mm'));
        dims.appendChild(document.createTextNode(' · Exhaust Ø '));
        dims.appendChild(node('span', 'agent-mono', fmt(r.exhaustDiaMm, 0) + ' mm'));
        card.appendChild(dims);

        if (checks.length) {
            var list = node('ul', 'agent-checks');
            checks.forEach(function (c) {
                var status = String(c.status || '').toLowerCase();
                var item = node('li', 'agent-check');
                item.appendChild(node('span', 'agent-pill ' + (status === 'pass' ? 'pass' : status === 'fail' ? 'fail' : 'warn'),
                    String(c.status || '').toUpperCase()));
                var text = node('span', null);
                text.appendChild(node('strong', null, (c.name || '') + ': '));
                text.appendChild(document.createTextNode(c.detail || ''));
                item.appendChild(text);
                list.appendChild(item);
            });
            card.appendChild(list);
        }

        var verdict = verdictOf(checks);
        card.appendChild(node('div', 'agent-verdict ' + verdict.cls, verdict.text));

        if (Array.isArray(r.assumptions) && r.assumptions.length) {
            var details = node('details', 'agent-assumptions');
            details.appendChild(node('summary', null, 'Assumptions used'));
            var ul = node('ul', null);
            r.assumptions.forEach(function (a) { ul.appendChild(node('li', null, a)); });
            details.appendChild(ul);
            card.appendChild(details);
        }

        if (showActions && entry.canSave) {
            var actions = node('div', 'agent-actions');

            var create = node('button', 'btn btn-primary btn-sm', 'Create design');
            create.type = 'button';
            create.addEventListener('click', openSavePanel);
            actions.appendChild(create);

            var edit = node('button', 'btn btn-outline-primary btn-sm', 'Edit inputs');
            edit.type = 'button';
            edit.addEventListener('click', function () {
                el.input.placeholder = 'e.g. Change the inlet line size to 8 inches';
                el.input.focus();
            });
            actions.appendChild(edit);

            card.appendChild(actions);
        }

        row.appendChild(card);
        return row;
    }

    function renderEntry(entry, showActions) {
        switch (entry.kind) {
            case 'user': return renderBubble('user', entry.text);
            case 'error': return renderBubble('error', entry.text);
            case 'result': return renderResult(entry, showActions);
            case 'link': return renderLink(entry);
            default: return renderBubble('bot', entry.text);
        }
    }

    function renderAll() {
        el.messages.textContent = '';
        el.messages.appendChild(renderBubble('bot', WELCOME));

        var lastResult = -1;
        state.entries.forEach(function (e, i) { if (e.kind === 'result') { lastResult = i; } });

        state.entries.forEach(function (e, i) {
            el.messages.appendChild(renderEntry(e, i === lastResult));
        });

        el.chips.hidden = state.entries.length > 0;
        renderDraft();
        scrollToBottom();
    }

    function addEntry(entry) {
        state.entries.push(entry);

        if (entry.kind === 'result') {
            var old = el.messages.querySelectorAll('.agent-actions');
            for (var i = 0; i < old.length; i++) { old[i].remove(); }
        }

        el.messages.appendChild(renderEntry(entry, true));
        el.chips.hidden = true;
        scrollToBottom();
        persist();
    }

    /* ── Typing indicator ─────────────────────────────────────────────── */

    function showTyping() {
        var row = node('div', 'agent-msg agent-msg-bot');
        var bubble = node('div', 'agent-bubble');
        var dots = node('span', 'agent-typing');
        dots.appendChild(node('span'));
        dots.appendChild(node('span'));
        dots.appendChild(node('span'));
        bubble.appendChild(dots);

        var note = node('span', 'agent-queued-note', '');
        bubble.appendChild(note);
        row.appendChild(bubble);
        el.messages.appendChild(row);
        scrollToBottom();

        var timer = setTimeout(function () {
            note.textContent = 'Waiting for the model…';
        }, QUEUE_NOTE_DELAY_MS);

        return function hide() {
            clearTimeout(timer);
            row.remove();
        };
    }

    /* ── Sending ──────────────────────────────────────────────────────── */

    function setBusy(busy) {
        state.busy = busy;
        el.send.disabled = busy;
        el.input.disabled = busy;
    }

    function autoResize() {
        el.input.style.height = 'auto';
        el.input.style.height = Math.min(el.input.scrollHeight, 120) + 'px';
    }

    function send(text) {
        var message = (text || '').trim();
        if (!message || state.busy) { return; }

        el.input.value = '';
        autoResize();
        addEntry({ kind: 'user', text: message });
        setBusy(true);
        var hideTyping = showTyping();

        request('POST', urls.message, { message: message, tabId: state.tabId })
            .then(function (res) {
                hideTyping();
                state.draft = res.draft || [];
                renderDraft();

                var hasResult = res.designComplete && res.result && !res.result.error;
                if (hasResult) {
                    addEntry({ kind: 'result', result: res.result, checks: res.checks || [], canSave: !!res.canSave });
                } else {
                    addEntry({ kind: 'bot', text: res.message });
                }
            })
            .catch(function (err) {
                hideTyping();
                addEntry({ kind: 'error', text: err.message || 'The assistant could not be reached.' });
            })
            .then(function () {
                setBusy(false);
                el.input.focus();
                persist();
            });
    }

    /* ── Reset ────────────────────────────────────────────────────────── */

    function resetChat() {
        if (state.busy) { return; }

        request('POST', urls.reset, { tabId: state.tabId })
            .catch(function () { /* local reset still applies */ })
            .then(function () {
                state.entries = [];
                state.draft = [];
                closeSavePanel();
                el.input.placeholder = 'Describe your cyclone requirement…';
                renderAll();
                persist();
            });
    }

    /* ── Save design ──────────────────────────────────────────────────── */

    function openSavePanel() {
        el.saveError.hidden = true;
        el.save.hidden = false;
        el.saveProject.textContent = '';

        var loading = node('option', null, 'Loading projects…');
        loading.value = '';
        el.saveProject.appendChild(loading);

        request('GET', urls.projects)
            .then(function (projects) {
                el.saveProject.textContent = '';

                if (!projects || !projects.length) {
                    var none = node('option', null, 'No projects available');
                    none.value = '';
                    el.saveProject.appendChild(none);
                    return;
                }

                projects.forEach(function (p) {
                    var label = (p.projectNumber ? p.projectNumber + ' — ' : '') + p.name;
                    var opt = node('option', null, label);
                    opt.value = String(p.id);
                    el.saveProject.appendChild(opt);
                });
                el.saveTag.focus();
            })
            .catch(function (err) {
                showSaveError(err.message);
            });
    }

    function closeSavePanel() {
        el.save.hidden = true;
        el.saveError.hidden = true;
        el.saveTag.value = '';
        el.saveName.value = '';
    }

    function showSaveError(message) {
        el.saveError.textContent = message;
        el.saveError.hidden = false;
    }

    function confirmSave() {
        var projectId = parseInt(el.saveProject.value, 10);
        var tag = el.saveTag.value.trim();

        if (!projectId) { showSaveError('Select a project.'); return; }
        if (!tag) { showSaveError('Enter a tag number.'); return; }

        el.saveConfirm.disabled = true;
        el.saveError.hidden = true;

        request('POST', urls.save, {
            projectId: projectId,
            tagNumber: tag,
            name: el.saveName.value.trim() || null,
            tabId: state.tabId
        })
            .then(function (res) {
                state.entries.forEach(function (e) { if (e.kind === 'result') { e.canSave = false; } });
                state.draft = [];
                closeSavePanel();
                state.entries.push({ kind: 'link', text: 'Design "' + tag + '" created.', url: res.resultsUrl });
                renderAll();
                persist();
            })
            .catch(function (err) {
                showSaveError(err.message);
            })
            .then(function () {
                el.saveConfirm.disabled = false;
            });
    }

    /* ── Events ───────────────────────────────────────────────────────── */

    el.launcher.addEventListener('click', openPanel);
    el.close.addEventListener('click', closePanel);
    el.reset.addEventListener('click', resetChat);
    el.send.addEventListener('click', function () { send(el.input.value); });

    el.input.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
            e.preventDefault();
            send(el.input.value);
        }
    });
    el.input.addEventListener('input', autoResize);

    el.chips.addEventListener('click', function (e) {
        var chip = e.target.closest('.agent-chip');
        if (chip) { send(chip.getAttribute('data-prompt')); }
    });

    el.draftToggle.addEventListener('click', function () {
        var expanded = el.draftToggle.getAttribute('aria-expanded') === 'true';
        el.draftToggle.setAttribute('aria-expanded', expanded ? 'false' : 'true');
    });

    el.saveCancel.addEventListener('click', closeSavePanel);
    el.saveConfirm.addEventListener('click', confirmSave);

    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape' && root.classList.contains('is-open')) { closePanel(); }
    });

    /* ── Init ─────────────────────────────────────────────────────────── */

    restore();
    renderAll();

    try {
        if (sessionStorage.getItem(OPEN_KEY) === '1') { openPanel(); }
    } catch (e) { /* ignore */ }
})();
