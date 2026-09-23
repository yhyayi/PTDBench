const LEADERBOARD_DATA = [
    {
        agent: 'ReasonAgent',
        framework: 'Tree-of-Thought + Reflection',
        model: 'Qwen3-235B-DAPO',
        params: '235B params · RL-tuned',
        scores: { math: 78.4, code: 82.1, reasoning: 71.3, science: 74.8, planning: 69.2 }
    },
    {
        agent: 'DeepAgent-R',
        framework: 'Multi-step Reasoning Chain',
        model: 'DeepSeek-R1',
        params: '671B MoE · Reasoning-tuned',
        scores: { math: 76.2, code: 79.8, reasoning: 73.1, science: 72.5, planning: 71.4 }
    },
    {
        agent: 'ToolFormer-X',
        framework: 'Tool-augmented ReAct',
        model: 'Llama-3-70B-RL',
        params: '70B params · GRPO-tuned',
        scores: { math: 71.5, code: 74.3, reasoning: 68.9, science: 70.1, planning: 66.8 }
    },
    {
        agent: 'PlanExec',
        framework: 'Plan-then-Execute pipeline',
        model: 'Qwen3-32B-GRPO',
        params: '32B params · RL-tuned',
        scores: { math: 69.8, code: 72.1, reasoning: 65.4, science: 67.3, planning: 72.6 }
    },
    {
        agent: 'Vanilla-CoT',
        framework: 'Chain-of-Thought baseline',
        model: 'Mistral-Large-RL',
        params: '123B params · SFT + RL',
        scores: { math: 65.3, code: 68.7, reasoning: 62.8, science: 64.9, planning: 61.5 }
    },
    {
        agent: 'ReasonAgent',
        framework: 'Tree-of-Thought + Reflection',
        model: 'DeepSeek-R1',
        params: '671B MoE · Reasoning-tuned',
        scores: { math: 77.1, code: 80.4, reasoning: 74.6, science: 73.2, planning: 68.9 }
    },
    {
        agent: 'ToolFormer-X',
        framework: 'Tool-augmented ReAct',
        model: 'Qwen3-235B-DAPO',
        params: '235B params · RL-tuned',
        scores: { math: 74.8, code: 79.2, reasoning: 70.1, science: 72.8, planning: 70.5 }
    }
];

const EVALUATION_DATA = {
    summary: [
        { value: '74.8', label: 'Mean hidden score', note: 'across published runs' },
        { value: '6', label: 'Domains covered', note: 'with a shared harness' },
        { value: '42m', label: 'Median run time', note: 'per agent evaluation' },
        { value: '18k', label: 'Median output tokens', note: 'per completed run' }
    ],
    domains: [
        { label: 'Math', score: 78.4 },
        { label: 'Code', score: 76.9 },
        { label: 'Reasoning', score: 73.8 },
        { label: 'Science', score: 71.6 },
        { label: 'Planning', score: 69.2 },
        { label: 'Language', score: 67.5 }
    ],
    resources: [
        { label: 'ReasonAgent', score: 75.2, runtime: '42m' },
        { label: 'DeepAgent-R', score: 74.8, runtime: '57m' },
        { label: 'ToolFormer-X', score: 73.5, runtime: '31m' },
        { label: 'PlanExec', score: 69.4, runtime: '26m' }
    ]
};

const PTDBENCH_TASKS = window.PTDBENCH_TASKS || [];

let currentTask = 'all';
let currentAgent = 'all';
let currentModel = 'all';

function getAvg(scores, task) {
    if (task === 'all') {
        const vals = Object.values(scores);
        return (vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(1);
    }
    return scores[task].toFixed(1);
}

function getScoreKeys(task) {
    if (task === 'all') return ['math', 'code', 'reasoning'];
    return [task];
}

function renderLeaderboard() {
    let filtered = LEADERBOARD_DATA.filter(row => {
        if (currentAgent !== 'all' && row.agent !== currentAgent) return false;
        if (currentModel !== 'all' && row.model !== currentModel) return false;
        return true;
    });

    filtered.sort((a, b) => {
        return parseFloat(getAvg(b.scores, currentTask)) - parseFloat(getAvg(a.scores, currentTask));
    });

    const list = document.querySelector('.leaderboard-list');
    const scoreKeys = getScoreKeys(currentTask);

    list.innerHTML = filtered.map((row, i) => `
        <div class="lb-row" data-agent="${row.agent}" data-model="${row.model}">
            <div class="lb-rank">${i + 1}</div>
            <div class="lb-agent">
                <span class="lb-agent-name">${row.agent}</span>
                <span class="lb-agent-framework">${row.framework}</span>
            </div>
            <div class="lb-model">
                <span class="lb-model-name">${row.model}</span>
                <span class="lb-model-params">${row.params}</span>
            </div>
            <div class="lb-scores">
                ${scoreKeys.map(key => `
                    <div class="lb-score-item">
                        <span class="lb-score-label">${key}</span>
                        <div class="lb-score-bar"><div class="lb-score-fill" style="width:${row.scores[key]}%"></div></div>
                    </div>
                `).join('')}
            </div>
            <div class="lb-avg">${getAvg(row.scores, currentTask)}</div>
        </div>
    `).join('');

    const headerScores = document.querySelector('.lb-header-scores');
    if (headerScores) {
        headerScores.textContent = currentTask === 'all' ? 'Scores' : currentTask.charAt(0).toUpperCase() + currentTask.slice(1);
    }
}

function initLeaderboard() {
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => {
                b.classList.remove('active');
                b.setAttribute('aria-pressed', 'false');
            });
            btn.classList.add('active');
            btn.setAttribute('aria-pressed', 'true');
            currentTask = btn.dataset.task;
            renderLeaderboard();
        });
    });

    const initialFilter = document.querySelector('.filter-btn.active');
    if (initialFilter) initialFilter.setAttribute('aria-pressed', 'true');

    const agentFilter = document.getElementById('agent-filter');
    const modelFilter = document.getElementById('model-filter');

    if (agentFilter) {
        agentFilter.addEventListener('change', () => {
            currentAgent = agentFilter.value;
            renderLeaderboard();
        });
    }

    if (modelFilter) {
        modelFilter.addEventListener('change', () => {
            currentModel = modelFilter.value;
            renderLeaderboard();
        });
    }

    renderLeaderboard();
}

function taskCard(task) {
    return `
        <a href="task.html?id=${encodeURIComponent(task.id)}" class="task-card">
            <span class="task-tag">${task.domain}</span>
            <h3>${task.title}</h3>
            <p>${task.summary}</p>
            <div class="task-meta">
                <span>${task.family}</span>
                <span>${task.failureType}</span>
            </div>
        </a>
    `;
}

function renderTaskCatalog() {
    const mount = document.getElementById('task-groups');
    if (!mount || !PTDBENCH_TASKS.length) return;

    const groups = PTDBENCH_TASKS.reduce((map, task) => {
        if (!map.has(task.family)) map.set(task.family, []);
        map.get(task.family).push(task);
        return map;
    }, new Map());

    mount.innerHTML = [...groups.entries()].map(([family, tasks], index) => `
        <details class="task-group" ${index === 0 ? 'open' : ''}>
            <summary>
                <span>${family}</span>
                <span class="task-group-count">${tasks.length} ${tasks.length === 1 ? 'task' : 'tasks'}</span>
            </summary>
            <div class="tasks-grid">${tasks.map(taskCard).join('')}</div>
        </details>
    `).join('');
}

function renderEvaluation() {
    const summary = document.getElementById('eval-summary');
    const domains = document.getElementById('domain-score-chart');
    const resources = document.getElementById('resource-chart');
    if (!summary || !domains || !resources) return;

    summary.innerHTML = EVALUATION_DATA.summary.map(item => `
        <div class="eval-stat">
            <strong>${item.value}</strong>
            <span>${item.label}</span>
            <small>${item.note}</small>
        </div>
    `).join('');

    domains.innerHTML = EVALUATION_DATA.domains.map(item => `
        <div class="chart-row">
            <span class="chart-label">${item.label}</span>
            <span class="chart-track"><i style="width:${item.score}%"></i></span>
            <strong class="chart-value">${item.score.toFixed(1)}</strong>
        </div>
    `).join('');

    resources.innerHTML = EVALUATION_DATA.resources.map(item => `
        <div class="chart-row">
            <span class="chart-label">${item.label}</span>
            <span class="chart-track"><i style="width:${item.score}%"></i></span>
            <strong class="chart-value">${item.score.toFixed(1)} <small>${item.runtime}</small></strong>
        </div>
    `).join('');
}

function initScrollMotion() {
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduceMotion) return;

    document.documentElement.classList.add('motion-ready');

    const sections = document.querySelectorAll('body > section:not(.hero), body > footer');
    sections.forEach(section => section.classList.add('scroll-reveal'));

    const revealAll = () => sections.forEach(section => section.classList.add('is-visible'));
    const hashId = window.location.hash.slice(1);
    const hashTarget = hashId ? document.getElementById(hashId) : null;
    if (hashTarget) hashTarget.classList.add('is-visible');

    requestAnimationFrame(() => {
        sections.forEach(section => {
            const rect = section.getBoundingClientRect();
            if (rect.top < window.innerHeight * 1.15 && rect.bottom > 0) {
                section.classList.add('is-visible');
            }
        });
    });

    if (!('IntersectionObserver' in window)) {
        sections.forEach(section => section.classList.add('is-visible'));
        return;
    }

    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            entry.target.classList.add('is-visible');
            observer.unobserve(entry.target);
        });
    }, {
        threshold: 0.08,
        rootMargin: '0px 0px -8% 0px'
    });

    sections.forEach(section => observer.observe(section));

    // Keep direct hash links and static hosts usable if observers are delayed.
    window.setTimeout(revealAll, 1200);
}

document.addEventListener('DOMContentLoaded', () => {
    renderTaskCatalog();
    initLeaderboard();
    renderEvaluation();
    initScrollMotion();
});
