(function () {
    'use strict';

    const tasks = window.PTDBENCH_TASKS || [];
    const instructions = window.PTDBENCH_INSTRUCTION_DATA || {};
    const params = new URLSearchParams(window.location.search);
    const task = tasks.find(item => item.id === params.get('id'));
    let activeMode = params.get('mode') === 'solution' ? 'solution' : 'issue';
    const mount = document.getElementById('task-detail-root');

    const resultSeeds = {
        issue: [
            { agent: 'Codex', model: 'GPT-5.5', score: 88.2, runtime: '18m' },
            { agent: 'Claude Code', model: 'Opus 4.1', score: 82.7, runtime: '24m' },
            { agent: 'OpenHands', model: 'DeepSeek-V3.1', score: 74.1, runtime: '31m' },
            { agent: 'Aider', model: 'Qwen3-Coder', score: 66.4, runtime: '16m' }
        ],
        solution: [
            { agent: 'Codex', model: 'GPT-5.5', score: 79.5, runtime: '96m' },
            { agent: 'Claude Code', model: 'Opus 4.1', score: 73.8, runtime: '112m' },
            { agent: 'OpenHands', model: 'DeepSeek-V3.1', score: 61.2, runtime: '138m' },
            { agent: 'Aider', model: 'Qwen3-Coder', score: 47.6, runtime: '84m' }
        ]
    };

    function renderParagraphs(paragraphs) {
        return paragraphs.map(paragraph => `<p>${paragraph}</p>`).join('');
    }

    function renderIssueInstruction(taskSpec = {}) {
        const issue = instructions.issue;
        const note = taskSpec.issueNote ? `<p>${taskSpec.issueNote}</p>` : '';
        return `
            <h1>${issue.title}</h1>
            ${renderParagraphs(issue.paragraphs)}
            ${note}
            <h2>Deliverable</h2>
            <p>Write one concise UTF-8 diagnosis to:</p>
            <pre><code>${issue.deliverablePath}</code></pre>
            <p>${issue.deliverable}</p>
        `;
    }

    function renderSolutionInstruction(taskSpec = {}) {
        const solution = instructions.solution;
        const launchNote = taskSpec.launchNote ? `<p>${taskSpec.launchNote}</p>` : '';
        const command = taskSpec.command
            ? `<p>Example:</p><pre><code>${taskSpec.command}</code></pre>`
            : '';
        const effectiveRun = (taskSpec.effectiveRun || [])
            .map(item => `<li><code>${item}</code></li>`)
            .join('');
        const solutionNote = taskSpec.solutionNote ? ` ${taskSpec.solutionNote}` : '';
        const strictLogs = taskSpec.strictLogs
            ? ' The run is scored only from files under $PTDBENCH_RESULTS/experiment_logs; logs written anywhere else are ignored.'
            : '';

        return `
            <h1>${solution.title}</h1>
            ${renderParagraphs(solution.paragraphs)}
            <p>If you find a defect, fix it in the staged workspace at <code>${taskSpec.workspace}</code>. Then run the experiment as a foreground process and wait until it returns. Do not use the network, <code>nohup</code>, <code>&amp;</code>, <code>tmux</code>, <code>screen</code>, or any detach mechanism. The dataset, held-out set, and model are fixed.${solutionNote}</p>
            <p>The launch script is:</p>
            <pre><code>${taskSpec.launch}</code></pre>
            ${launchNote}
            ${command}
            <h2>Effective run</h2>
            <p>The effective run must use:</p>
            <ul>${effectiveRun}</ul>
            <p>Create <code>$PTDBENCH_RESULTS/experiment_logs</code> first. Write the training and evaluation log into that directory. Offline W&amp;B data must be created below <code>$PTDBENCH_RESULTS/experiment_logs/wandb</code>.${strictLogs} Do not copy, edit, or fabricate log metrics. Otherwise the run is scored 0.</p>
        `;
    }

    function previewResults(level) {
        const offset = (tasks.indexOf(task) % 4) * 1.4;
        return resultSeeds[level].map((result, index) => ({
            ...result,
            score: Math.max(0, Math.min(100, result.score - offset + index * 0.35))
        }));
    }

    function renderResults(level) {
        const results = previewResults(level);
        const ordered = [...results].map(result => result.score).sort((a, b) => a - b);
        const median = (ordered[1] + ordered[2]) / 2;
        const passLine = level === 'issue' ? 80 : 70;
        const passCount = results.filter(result => result.score >= passLine).length;
        const rows = results.map((result, index) => {
            const status = result.score >= passLine ? 'Pass' : 'Below pass line';
            return `
                <div class="agent-score-row ${index === 0 ? 'is-leading' : ''}">
                    <span class="agent-score-rank">${String(index + 1).padStart(2, '0')}</span>
                    <div class="agent-score-identity"><strong>${result.agent}</strong><span>${result.model}</span></div>
                    <div class="agent-score-visual"><i><b style="width:${result.score}%"></b></i></div>
                    <strong class="agent-score-value">${result.score.toFixed(1)}</strong>
                    <span class="agent-score-runtime">${result.runtime}</span>
                    <span class="agent-score-status ${result.score >= passLine ? 'is-pass' : ''}">${status}</span>
                </div>
            `;
        }).join('');

        return `
            <section class="mode-results">
                <div class="mode-section-heading">
                    <div><span class="task-section-kicker">Agent results</span><h2>${level === 'issue' ? 'Diagnosis performance' : 'Repair performance'}</h2></div>
                    <span class="preview-badge">Preview data</span>
                </div>
                <div class="result-summary-line">
                    <div class="result-summary-primary"><strong>${results[0].score.toFixed(1)}</strong><span>Best score</span></div>
                    <div><strong>${median.toFixed(1)}</strong><span>Median score</span></div>
                    <div><strong>${results.length}</strong><span>Agent runs</span></div>
                    <div><strong>${passCount}/${results.length}</strong><span>Passed</span></div>
                </div>
                <div class="agent-score-table">
                    <div class="agent-score-row agent-score-head"><span>#</span><span>Agent / model</span><span>Score distribution</span><span>Score</span><span>Runtime</span><span>Result</span></div>
                    ${rows}
                </div>
                <p class="preview-note">Illustrative values for layout review. Replace them with published benchmark runs in task-data.js.</p>
            </section>
        `;
    }

    function renderMode(level, taskSpec) {
        const isIssue = level === 'issue';
        return `
            <div class="task-mode-panel" data-mode-panel="${level}" ${level === activeMode ? '' : 'hidden'}>
                <section class="mode-instruction">
                    <div class="mode-section-heading">
                        <div><span class="task-section-kicker">${isIssue ? 'Issue evaluation' : 'Solution evaluation'}</span><h2>Task instruction</h2></div>
                        <span class="instruction-file">evaluations/${level}/instruction.md</span>
                    </div>
                    <article class="instruction-document">
                        <div class="instruction-document-label"><span>instruction.md</span><span>This is what the agent is given</span></div>
                        <div class="instruction-document-body">
                            ${isIssue ? renderIssueInstruction(taskSpec) : renderSolutionInstruction(taskSpec)}
                        </div>
                        <div class="verifier-line">
                            <div><span>Verifier</span><p>${isIssue ? 'The diagnosis is compared with a hidden reference by the task judge.' : 'The repaired run is graded from fresh outputs by a metric-only verifier.'}</p></div>
                            <div><span>Primary metric</span><code>${task.metric}</code></div>
                        </div>
                    </article>
                </section>
                ${renderResults(level)}
            </div>
        `;
    }

    function render() {
        if (!mount) return;
        if (!task) {
            mount.innerHTML = '<div class="task-not-found"><h1>Task not found</h1><p>Return to the <a href="index.html#tasks">task catalog</a>.</p></div>';
            return;
        }

        const taskSpec = instructions.tasks[task.id];
        document.title = `${task.title} — DataCall`;
        mount.innerHTML = `
            <div class="task-breadcrumb"><a href="index.html#tasks">Tasks</a><span>/</span><span>${task.family}</span></div>
            <header class="task-page-header">
                <div class="task-header-tags"><span>${task.domain}</span><span>${task.failureType}</span></div>
                <h1>${task.title}</h1>
                <p>${task.summary}</p>
                <div class="task-settings" aria-label="Task settings">
                    <div><span>Model</span><strong>${task.model}</strong></div>
                    <div><span>Dataset</span><strong>${task.dataset}</strong></div>
                    <div><span>Framework</span><strong>${task.framework} · ${task.algorithm}</strong></div>
                    <div><span>Metric</span><strong>${task.metric}</strong></div>
                </div>
            </header>

            <div class="task-mode-switch" role="tablist" aria-label="Task mode">
                <button class="task-mode-button ${activeMode === 'issue' ? 'is-active' : ''}" type="button" role="tab" aria-selected="${activeMode === 'issue'}" data-mode-tab="issue"><span>Issue</span><small>Diagnose the experiment</small></button>
                <button class="task-mode-button ${activeMode === 'solution' ? 'is-active' : ''}" type="button" role="tab" aria-selected="${activeMode === 'solution'}" data-mode-tab="solution"><span>Solution</span><small>Repair and rerun</small></button>
            </div>

            ${renderMode('issue', taskSpec)}
            ${renderMode('solution', taskSpec)}
        `;

        mount.querySelectorAll('[data-mode-tab]').forEach(tab => {
            tab.addEventListener('click', () => {
                const level = tab.dataset.modeTab;
                activeMode = level;
                mount.querySelectorAll('[data-mode-tab]').forEach(item => {
                    const active = item === tab;
                    item.classList.toggle('is-active', active);
                    item.setAttribute('aria-selected', String(active));
                });
                mount.querySelectorAll('[data-mode-panel]').forEach(panel => {
                    panel.hidden = panel.dataset.modePanel !== level;
                });
                const url = new URL(window.location.href);
                url.searchParams.set('mode', level);
                window.history.replaceState({}, '', url);
            });
        });
    }

    document.addEventListener('DOMContentLoaded', render);
})();
