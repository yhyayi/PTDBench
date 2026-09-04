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
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentTask = btn.dataset.task;
            renderLeaderboard();
        });
    });

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

document.addEventListener('DOMContentLoaded', initLeaderboard);
