document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('recognition-form');
    const resultsList = document.getElementById('results-list');
    const statusBadge = document.getElementById('status-badge');
    const uptimeDisplay = document.getElementById('uptime-timer');

    let sequenceInterval;
    let startTime = Date.now();

    // Uptime Timer
    setInterval(() => {
        const diff = Date.now() - startTime;
        const h = Math.floor(diff / 3600000).toString().padStart(2, '0');
        const m = Math.floor((diff % 3600000) / 60000).toString().padStart(2, '0');
        const s = Math.floor((diff % 60000) / 1000).toString().padStart(2, '0');
        uptimeDisplay.textContent = `${h}:${m}:${s}`;
    }, 1000);

    const PROTOCOLS = {
        'NO': { color: [0, 255, 255], name: 'NORMAL_GAIT', impact: '0%', desc: 'No interference detected. Baseline neural signature.' },
        'CA': { color: [255, 0, 255], name: 'CARRYING_BAG', impact: '18.4%', desc: 'Foreign mass detected in peripheral gait cycle. Lower limb occlusion.' },
        'CO': { color: [0, 255, 0], name: 'CROWD_OVERLAP', impact: '34.2%', desc: 'Partial torso and limb occlusion due to multiple dynamic subjects.' },
        'ST': { color: [255, 255, 0], name: 'STATIC_OBJECT', impact: '22.8%', desc: 'Fixed obstacle obstructing gait path. Pattern fragmentation.' }
    };

    let currentInterval = null;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Reset state
        if (currentInterval) clearInterval(currentInterval);
        resultsList.innerHTML = '<div class="flex items-center justify-center py-20 text-cyan-400 animate-pulse font-mono text-xs tracking-widest">INITIALIZING_SCAN...</div>';
        statusBadge.textContent = 'SCANNING...';
        statusBadge.classList.add('text-fuchsia-500');

        const formData = new FormData(form);
        const occType = formData.get('occ_type') || 'NO';
        
        try {
            const response = await fetch('/api/recognize', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.error) {
                statusBadge.textContent = 'SCAN_ERROR';
                resultsList.innerHTML = `<div class="text-red-500 font-mono text-[10px] p-4 border border-red-900 bg-red-900/10">${data.error}</div>`;
                return;
            }

            // Display Results
            displayResults(data.top_matches);
            
            // Show occlusion image for the selected protocol
            showOcclusionImage(data.occ_type || occType);
            
            statusBadge.textContent = 'MATCH_FOUND';
            statusBadge.classList.remove('text-fuchsia-500');
            statusBadge.classList.add('text-cyan-400');

        } catch (error) {
            console.error('Error:', error);
            statusBadge.textContent = 'LINK_FAILURE';
            resultsList.innerHTML = '<div class="text-red-500 font-mono text-[10px] p-4">SYSTEM_OFFLINE</div>';
        }
    });

    // Maps each protocol to its pre-rendered high-quality image.
    // Images are in backend/static/images/occ_*.png
    function showOcclusionImage(occType) {
        const protocol = PROTOCOLS[occType] || PROTOCOLS['NO'];

        // Update protocol log below the viewport
        const log = document.getElementById('protocol-log');
        log.classList.remove('hidden');
        document.getElementById('log-protocol').innerText = protocol.name;
        document.getElementById('log-impact').innerText = protocol.impact;
        document.getElementById('log-desc').innerText = protocol.desc;

        // Swap placeholder for the image
        const placeholder = document.getElementById('occ-placeholder');
        const wrap = document.getElementById('occ-image-wrap');
        const img  = document.getElementById('occ-image');

        if (placeholder) placeholder.classList.add('hidden');
        wrap.classList.remove('hidden');
        wrap.style.display = 'flex';

        // Brief fade-in effect
        img.style.opacity = '0';
        img.src = `/static/images/occ_${occType}.png`;
        img.onload = () => {
            img.style.transition = 'opacity 0.4s ease';
            img.style.opacity = '1';
        };
    }


    function displayResults(matches) {
        resultsList.innerHTML = '';
        
        matches.forEach((match, index) => {
            const confidence = (match.confidence * 100).toFixed(1);
            const isTop = index === 0;
            const colorClass = isTop ? 'text-cyan-400' : 'text-zinc-400';
            const barClass = isTop ? 'bg-cyan-400 shadow-[0_0_8px_rgba(0,255,255,0.5)]' : 'bg-zinc-700';
            const riskText = isTop ? 'AUTHENTICATED' : 'PROBABLE';
            const riskColor = isTop ? 'text-cyan-400' : 'text-zinc-500';

            const item = document.createElement('div');
            item.className = 'bg-surface-container p-4 border border-white/5 hover:border-cyan-400/30 transition-all group bracket-corners mb-2';
            item.innerHTML = `
                <div class="flex justify-between items-start mb-2">
                    <div class="flex flex-col">
                        <span class="text-xs font-mono font-bold text-white tracking-wider">${match.identity.replace('_', ' ')}</span>
                        <span class="text-[9px] font-mono text-zinc-500 uppercase">SIGNAL_STRENGTH: ${confidence}%</span>
                    </div>
                    <span class="${riskColor} font-mono text-[9px] font-bold tracking-widest">${riskText}</span>
                </div>
                <div class="h-1 w-full bg-zinc-900/50 neon-progress">
                    <div class="h-full ${barClass} transition-all duration-1000 ease-out" style="width: 0%"></div>
                </div>
            `;
            resultsList.appendChild(item);

            // Animate progress bar
            setTimeout(() => {
                item.querySelector('.h-full').style.width = `${confidence}%`;
            }, 100);
        });
    }
});
