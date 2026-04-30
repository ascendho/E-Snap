document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const accessCodeInput = document.getElementById('access-code');
    const saveKeyBtn = document.getElementById('save-key-btn');
    const togglePwdBtn = document.getElementById('toggle-pwd-btn');
    const accessInputWrapper = document.getElementById('access-input-wrapper');
    const verifiedBadge = document.getElementById('verified-badge');
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const themeToggleLightIcon = document.getElementById('theme-toggle-light-icon');
    const themeToggleDarkIcon = document.getElementById('theme-toggle-dark-icon');
    const quickActions = document.getElementById('quick-actions');
    const quickPills = document.querySelectorAll('.quick-pill');

    // Automatically adapt to current host domain for full-stack deployment
    const CHAT_API_URL = '/chat/stream';
    const VALIDATE_URL = '/validate';
    const HEALTH_URL = '/health';
    let isRequestLocked = false;

    const setRequestLocked = (locked) => {
        isRequestLocked = locked;
        sendBtn.disabled = locked;
        userInput.disabled = locked;
        quickPills.forEach((pill) => {
            pill.classList.toggle('pointer-events-none', locked);
            pill.classList.toggle('opacity-50', locked);
            pill.setAttribute('aria-disabled', locked ? 'true' : 'false');
        });
    };

    // Theme Management
    if (localStorage.getItem('color-theme') === 'dark') {
        document.documentElement.classList.add('dark');
        themeToggleLightIcon.classList.remove('hidden');
    } else {
        document.documentElement.classList.remove('dark');
        themeToggleDarkIcon.classList.remove('hidden');
    }

    themeToggleBtn.addEventListener('click', () => {
        themeToggleDarkIcon.classList.toggle('hidden');
        themeToggleLightIcon.classList.toggle('hidden');
        if (localStorage.getItem('color-theme')) {
            if (localStorage.getItem('color-theme') === 'light') {
                document.documentElement.classList.add('dark');
                localStorage.setItem('color-theme', 'dark');
            } else {
                document.documentElement.classList.remove('dark');
                localStorage.setItem('color-theme', 'light');
            }
        } else {
            if (document.documentElement.classList.contains('dark')) {
                document.documentElement.classList.remove('dark');
                localStorage.setItem('color-theme', 'light');
            } else {
                document.documentElement.classList.add('dark');
                localStorage.setItem('color-theme', 'dark');
            }
        }
    });

    // Password Eye Toggle
    togglePwdBtn.addEventListener('click', () => {
        const type = accessCodeInput.getAttribute('type') === 'password' ? 'text' : 'password';
        accessCodeInput.setAttribute('type', type);
        // Switch SVG based on type
        if(type === 'text') {
            togglePwdBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4 text-gray-800 dark:text-gray-200"><path stroke-linecap="round" stroke-linejoin="round" d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 7.894L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 10-4.243-4.243m4.242 4.242L9.88 9.88" /></svg>';
        } else {
            togglePwdBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4"><path stroke-linecap="round" stroke-linejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" /><path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>';
        }
    });

    const setSuccessUI = (key) => {
        finalAccessKey = key;
        localStorage.setItem('agent_access_key', key);
        accessInputWrapper.classList.add('hidden');
        saveKeyBtn.classList.add('hidden');
        verifiedBadge.classList.replace('hidden', 'flex');
    };

    const setEditingUI = () => {
        accessInputWrapper.classList.remove('hidden');
        saveKeyBtn.classList.remove('hidden');
        verifiedBadge.classList.replace('flex', 'hidden');
        accessCodeInput.value = finalAccessKey;
        accessCodeInput.focus();
    };

    // Access Key Persistence
    let finalAccessKey = localStorage.getItem('agent_access_key') || '';
    if (finalAccessKey) {
        setSuccessUI(finalAccessKey);
    }

    verifiedBadge.addEventListener('click', setEditingUI);

    const validateAccessCode = async () => {
        const val = accessCodeInput.value.trim();
        if (!val) return;

        const prevText = saveKeyBtn.innerHTML;
        saveKeyBtn.innerHTML = '<svg class="animate-spin -ml-1 mr-2 w-4 h-4 text-white inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>保存';
        saveKeyBtn.disabled = true;
        saveKeyBtn.classList.add('opacity-80', 'cursor-wait');
        accessCodeInput.disabled = true;
        
        try {
            const response = await fetch(VALIDATE_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ access_code: val })
            });
            
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || '验证失败');
            }
            
            // Save mode
            setSuccessUI(val);
        } catch(error) {
            saveKeyBtn.innerHTML = prevText;
            alert(error.message);
            accessCodeInput.value = '';
            accessCodeInput.disabled = false;
            accessCodeInput.focus();
        } finally {
            saveKeyBtn.classList.remove('opacity-80', 'cursor-wait');
            saveKeyBtn.disabled = false;
            accessCodeInput.disabled = false;
        }
    };

    saveKeyBtn.addEventListener('click', validateAccessCode);
    
    // Auto validate on Enter or Blur
    accessCodeInput.addEventListener('keypress', (e) => {
        if(e.key === 'Enter') validateAccessCode();
    });
    
    // Prevent togglePwdBtn from stealing focus on click
    togglePwdBtn.addEventListener('mousedown', (e) => {
        e.preventDefault();
    });

    accessCodeInput.addEventListener('blur', () => {
        // Delay blur to allow save-key-btn click without immediately alerting block
        setTimeout(() => {
            if(document.activeElement !== saveKeyBtn && document.activeElement !== togglePwdBtn && accessCodeInput.value.trim() && !verifiedBadge.classList.contains('flex')) {
                validateAccessCode();
            }
        }, 100);
    });

    // Handle quick pills (Persistent)
    quickPills.forEach(pill => {
        pill.addEventListener('click', (e) => {
            if (isRequestLocked) return;
            const presetTxt = e.currentTarget.lastElementChild ? e.currentTarget.lastElementChild.textContent.trim() : e.currentTarget.textContent.trim();
            handleSend(presetTxt);
        });
    });

    // Helper: Scroll to bottom
    const scrollToBottom = () => {
        chatBox.scrollTop = chatBox.scrollHeight;
    };

    // Helper: Escape HTML
    const escapeHTML = (str) => {
        return str.replace(/[&<>'"]/g, 
            tag => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                "'": '&#39;',
                '"': '&quot;'
            }[tag] || tag)
        );
    };

    // Helper: Format Server Response using marked.js
    const formatText = (text) => {
        // use marked if available, fallback to simple formatting
        if (typeof marked !== 'undefined') {
            return marked.parse(text);
        }
        return escapeHTML(text).replace(/\n/g, '<br/>');
    };

    const BADGE_CONFIG = {
        zero_intercept: {
            text: '🛡️ Zero-Layer Intercept',
            classes: 'bg-yellow-50 dark:bg-yellow-500/10 text-yellow-700 dark:text-yellow-400 border border-yellow-200 dark:border-yellow-500/20 rounded-full font-medium tracking-wide shadow-sm transition-colors'
        },
        cache_exact: {
            text: '⚡ Exact Cache Hit',
            classes: 'bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-500/20 rounded-full font-medium tracking-wide transition-colors'
        },
        cache_near_exact: {
            text: '✨ Near-Exact Cache Hit',
            classes: 'bg-teal-50 dark:bg-teal-500/10 text-teal-700 dark:text-teal-400 border border-teal-200 dark:border-teal-500/20 rounded-full font-medium tracking-wide transition-colors'
        },
        cache_semantic_reuse: {
            text: '🧠 Reranked Cache Reuse',
            classes: 'bg-green-50 dark:bg-green-500/10 text-green-700 dark:text-green-400 border border-green-200 dark:border-green-500/20 rounded-full font-medium tracking-wide transition-colors'
        },
        cache_partial_reuse: {
            text: '🧩 Partial Cache Reuse + RAG',
            classes: 'bg-indigo-50 dark:bg-indigo-500/10 text-indigo-700 dark:text-indigo-400 border border-indigo-200 dark:border-indigo-500/20 rounded-full font-medium tracking-wide transition-colors'
        },
        rag_full_research: {
            text: '🔎 LLM Analysis / Full RAG',
            classes: 'bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-500/20 rounded-full font-medium tracking-wide transition-colors'
        }
    };

    const resolveLabelMeta = (meta = {}) => {
        let fallbackLabelKey = 'rag_full_research';
        if (meta.intercepted) {
            fallbackLabelKey = 'zero_intercept';
        } else if (meta.cache_match_type === 'exact') {
            fallbackLabelKey = 'cache_exact';
        } else if (meta.cache_match_type === 'near_exact') {
            fallbackLabelKey = 'cache_near_exact';
        } else if (meta.cache_reuse_mode === 'partial_reuse') {
            fallbackLabelKey = 'cache_partial_reuse';
        } else if (meta.cache_reuse_mode === 'full_reuse') {
            fallbackLabelKey = 'cache_semantic_reuse';
        }
        const labelKey = meta.label_key || fallbackLabelKey;
        return BADGE_CONFIG[labelKey] || BADGE_CONFIG.rag_full_research;
    };

    window.copyToClipboard = (text, button) => {
        navigator.clipboard.writeText(text).then(() => {
            const old = button.innerHTML;
            button.innerHTML = '✅ 已复制！';
            setTimeout(() => button.innerHTML = old, 1500);
        });
    };

    window.editMessage = (text) => {
        userInput.value = text;
        userInput.focus();
    };

    // Helper: Create User Message Element
    const appendUserMessage = (text) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'flex w-full mt-4 space-x-4 max-w-2xl ml-auto justify-end group transition-colors duration-300';
        const escapedText = escapeHTML(text);
        const safeJsText = escapedText.replace(/'/g, "\\'").replace(/\n/g, "\\n");
        msgDiv.innerHTML = '<div class="flex flex-col items-end w-full">' +
            '<div class="flex space-x-2 mb-1 opacity-0 group-hover:opacity-100 transition-opacity text-gray-400 dark:text-gray-500 text-xs">' +
                '<button class="hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 px-1.5 py-0.5 rounded cursor-pointer transition-colors" onclick="window.copyToClipboard(\'' + safeJsText + '\', this)">📋 复制</button>' +
                '<button class="hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 px-1.5 py-0.5 rounded cursor-pointer transition-colors" onclick="window.editMessage(\'' + safeJsText + '\')">✏️ 编辑</button>' +
            '</div>' +
            '<div class="bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100 border border-gray-200/60 dark:border-gray-600 p-4 rounded-2xl rounded-tr-sm shadow-sm text-[15px] break-words whitespace-pre-wrap leading-relaxed inline-block max-w-full transition-colors duration-300">' + escapedText + '</div>' +
            '</div>' +
            '<div class="flex-shrink-0 mt-6 h-9 w-9 rounded-full bg-gradient-to-tr from-emerald-100 to-teal-50 dark:from-emerald-900/40 dark:to-teal-900/30 flex items-center justify-center shadow-sm transition-colors duration-300 ring-1 ring-emerald-200/60 dark:ring-teal-700/40">' +
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="w-5 h-5 text-emerald-600 dark:text-teal-400">' +
                  '<circle cx="12" cy="12" r="10"></circle>' +
                  '<path d="M8 14s1.5 2 4 2 4-2 4-2"></path>' +
                  '<line x1="9" y1="9" x2="9.01" y2="9"></line>' +
                  '<line x1="15" y1="9" x2="15.01" y2="9"></line>' +
                '</svg>' +
            '</div>';
        chatBox.appendChild(msgDiv);
        scrollToBottom();
    };

    // Helper: Create Bot Message Element
    const buildBadgeHtml = (meta = {}) => {
        const latency = meta.latency_ms ?? meta.latency ?? null;
        if (!latency) return '';

        const labelMeta = resolveLabelMeta(meta);
        const badgeStyle = '<span class="px-2.5 py-1 ' + labelMeta.classes + '">' + labelMeta.text + '</span>';

        return '<div class="mt-2 flex items-center space-x-2 text-[11px] uppercase">' +
            badgeStyle +
            '<span class="text-gray-400 dark:text-gray-500 font-medium tracking-wider">⏱ ' + latency + 'ms</span>' +
        '</div>';
    };

    const appendBotMessage = (text, meta = {}) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'flex w-full mt-4 space-x-4 max-w-2xl group transition-colors duration-300';
        const badgeHtml = buildBadgeHtml(meta);
        const borderStyle = meta.intercepted ? 'border-yellow-200/60 dark:border-yellow-500/30' : 'border-gray-100 dark:border-gray-700';

        const safeJsText = escapeHTML(text).replace(/'/g, "\\'").replace(/\n/g, "\\n");
        msgDiv.innerHTML = '<div class="flex-shrink-0 mt-6 h-9 w-9 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center shadow-md">' +
                '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5 text-white"><path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09l2.846.813-2.846.813a4.5 4.5 0 00-3.09 3.09z" /></svg>' +
            '</div>' +
            '<div class="flex flex-col flex-1 mr-8">' +
                '<div class="flex space-x-2 mb-1 opacity-0 group-hover:opacity-100 transition-opacity text-gray-400 dark:text-gray-500 text-xs">' +
                    '<button class="hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 px-1.5 py-0.5 rounded cursor-pointer transition-colors" onclick="window.copyToClipboard(\'' + safeJsText + '\', this)">📋 复制</button>' +
                '</div>' +
                '<div class="bg-white dark:bg-gray-800 p-4 rounded-2xl rounded-tl-sm shadow-sm border ' + borderStyle + ' text-gray-800 dark:text-gray-200 text-[15px] whitespace-normal leading-relaxed model-response font-medium transition-colors duration-300 markdown-content overflow-hidden">' +
                    formatText(text) +
                '</div>' +
                badgeHtml +
            '</div>';
        
        chatBox.appendChild(msgDiv);
        scrollToBottom();
    };

    const createStreamingBotMessage = () => {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'flex w-full mt-4 space-x-4 max-w-2xl group transition-colors duration-300';
        msgDiv.innerHTML = '<div class="flex-shrink-0 mt-6 h-9 w-9 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center shadow-md">' +
                '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5 text-white"><path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09l2.846.813-2.846.813a4.5 4.5 0 00-3.09 3.09z" /></svg>' +
            '</div>' +
            '<div class="flex flex-col flex-1 mr-8">' +
                '<div class="flex space-x-2 mb-1 opacity-0 group-hover:opacity-100 transition-opacity text-gray-400 dark:text-gray-500 text-xs streaming-actions"></div>' +
                '<div class="bg-white dark:bg-gray-800 p-4 rounded-2xl rounded-tl-sm shadow-sm border border-gray-100 dark:border-gray-700 text-gray-800 dark:text-gray-200 text-[15px] whitespace-pre-wrap leading-relaxed model-response font-medium transition-colors duration-300 markdown-content overflow-hidden streaming-body"></div>' +
                '<div class="streaming-badge"></div>' +
            '</div>';

        chatBox.appendChild(msgDiv);
        scrollToBottom();
        return {
            root: msgDiv,
            body: msgDiv.querySelector('.streaming-body'),
            badge: msgDiv.querySelector('.streaming-badge'),
            actions: msgDiv.querySelector('.streaming-actions'),
            rawText: ''
        };
    };

    const appendStreamingChunk = (streamMessage, chunk) => {
        streamMessage.rawText += chunk;
        streamMessage.body.textContent = streamMessage.rawText;
        scrollToBottom();
    };

    const finalizeStreamingBotMessage = (streamMessage, meta) => {
        const finalText = streamMessage.rawText || meta.answer || '';
        const safeJsText = escapeHTML(finalText).replace(/'/g, "\\'").replace(/\n/g, "\\n");
        streamMessage.body.innerHTML = formatText(finalText);
        streamMessage.actions.innerHTML = '<button class="hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 px-1.5 py-0.5 rounded cursor-pointer transition-colors" onclick="window.copyToClipboard(\'' + safeJsText + '\', this)">📋 复制</button>';
        streamMessage.badge.innerHTML = buildBadgeHtml(meta);
        scrollToBottom();
    };

    // Helper: Create Error Message Element with Retry
    const appendErrorMessage = (message, failedQuery) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'flex w-full mt-4 space-x-4 max-w-2xl transition-colors duration-300';
        msgDiv.innerHTML = `
            <div class="flex-shrink-0 mt-1 h-9 w-9 rounded-full bg-red-50 dark:bg-red-500/10 border border-red-100 dark:border-red-500/20 flex items-center justify-center shadow-sm">
                <span class="text-red-500 font-bold text-sm">!</span>
            </div>
            <div class="flex flex-col flex-1 mr-8">
                <div class="bg-red-50 dark:bg-red-900/20 p-4 rounded-2xl rounded-tl-sm border border-red-100/60 dark:border-red-800/30 text-red-800 dark:text-red-200 text-[15px] leading-relaxed shadow-sm transition-colors duration-300">
                    <p class="font-medium mb-2">${escapeHTML(message)}</p>
                    <button onclick="window.retryLastRequest(this)" class="inline-flex items-center space-x-1 text-sm bg-white dark:bg-red-950/40 hover:bg-gray-50 dark:hover:bg-red-900/60 border border-red-200 dark:border-red-800/50 text-red-600 dark:text-red-400 px-3 py-1.5 rounded-lg transition-colors shadow-sm focus:outline-none">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-4 h-4"><path stroke-linecap="round" stroke-linejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" /></svg>
                        <span>重试 (Retry)</span>
                    </button>
                </div>
            </div>
        `;
        chatBox.appendChild(msgDiv);
        scrollToBottom();
    };

    // Helper: Create Loading Indicator
    const createLoadingIndicator = () => {
        const id = 'loader-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.className = 'flex w-full mt-4 space-x-4 max-w-2xl transition-colors duration-300';
        msgDiv.id = id;
        msgDiv.innerHTML = `
            <div class="flex-shrink-0 mt-1 h-9 w-9 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center shadow-md">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5 text-white">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09l2.846.813-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
                </svg>
            </div>
            <div class="flex flex-col flex-1 mr-8">
                <div class="p-4 py-5 flex items-center space-x-2 transition-colors duration-300 text-indigo-500 dark:text-purple-400">
                    <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5 animate-spin" viewBox="0 0 24 24" fill="url(#sparkle-gradient)" stroke="none">
                      <defs>
                        <linearGradient id="sparkle-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                          <stop offset="0%" stop-color="#8b5cf6" />
                          <stop offset="100%" stop-color="#3b82f6" />
                        </linearGradient>
                      </defs>
                      <path d="M12 2L14.6 9.4L22 12L14.6 14.6L12 22L9.4 14.6L2 12L9.4 9.4L12 2Z" />
                    </svg>
                    <span class="text-[15px] font-medium tracking-wide animate-pulse bg-clip-text text-transparent bg-gradient-to-r from-indigo-500 to-purple-600 dark:from-indigo-400 dark:to-purple-500">
                        <span class="loader-text">Thinking...</span>
                    </span>
                </div>
            </div>
        `;
        chatBox.appendChild(msgDiv);
        scrollToBottom();
        return id;
    };

    const removeLoadingIndicator = (id) => {
        const el = document.getElementById(id);
        if (el) el.remove();
    };

    const updateLoadingIndicator = (id, text) => {
        const el = document.getElementById(id);
        if (!el) return;
        const textEl = el.querySelector('.loader-text');
        if (textEl) {
            textEl.textContent = text;
        }
    };

    // Retry global binding
    window.lastAttemptedQuery = "";
    window.retryLastRequest = (btnElement) => {
        if (isRequestLocked) return;
        if(window.lastAttemptedQuery) {
            if (btnElement) {
                const errorBox = btnElement.closest('.flex.w-full.mt-4');
                if (errorBox) errorBox.remove();
            }
            handleSend(window.lastAttemptedQuery, true);
        }
    };

    const ensureBackendReady = async () => {
        try {
            const response = await fetch(HEALTH_URL, { cache: 'no-store' });
            if (!response.ok) {
                throw new Error('health-check-failed');
            }

            const health = await response.json();
            if (!health.ready) {
                return {
                    ok: false,
                    message: health.message || "后端正在初始化，请等待终端出现 Application startup complete 后再试。"
                };
            }

            return { ok: true };
        } catch (error) {
            return {
                ok: false,
                message: "网络请求失败，后端服务可能未启动或网络异常。"
            };
        }
    };

    // Action: Handle Send
    const handleSend = async (overrideText = null, isRetry = false) => {
        const query = typeof overrideText === 'string' ? overrideText : userInput.value.trim();
        const accessCode = finalAccessKey || accessCodeInput.value.trim();

        if (!query || isRequestLocked) return;

        if (!accessCode) {
            alert('🔒 请先在右上角输入专属【邀请码】并点击【保存】！\n💡 您可以在我的简历中找到它。');
            accessCodeInput.focus();
            return;
        }

        let loaderId = null;
        setRequestLocked(true);

        try {
            window.lastAttemptedQuery = query; // Save for retry even when backend is not ready yet

            const readiness = await ensureBackendReady();
            if (!readiness.ok) {
                appendErrorMessage(readiness.message, query);
                return;
            }

            // Add user msg if not retrying
            if(!isRetry) {
                 appendUserMessage(query);
            }
            
            if (typeof overrideText !== 'string') {
                userInput.value = '';
            }
            
            // Show loader
            loaderId = createLoadingIndicator();

            // Request backend
            const response = await fetch(CHAT_API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    access_code: accessCode
                })
            });

            if (!response.ok) {
                if (response.status === 401) {
                    throw new Error('❌ 邀请码错误！请求被拦截，未消耗 Tokens。');
                } else if (response.status === 429) {
                    throw new Error('⚠️ API 访问过于频繁。防刷限制已触发，请稍后再试。');
                } else if (response.status === 503) {
                    const errData = await response.json();
                    throw new Error(errData.detail || '后端正在初始化，请稍后重试。');
                }
                const errData = await response.json();
                throw new Error(errData.detail || 'The API request failed.');
            }

            if (!response.body) {
                throw new Error('Streaming response body is unavailable.');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let streamMessage = null;

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (!line.trim()) continue;
                    const event = JSON.parse(line);

                    if (event.type === 'status') {
                        updateLoadingIndicator(loaderId, event.message || 'Thinking...');
                        continue;
                    }

                    if (event.type === 'token') {
                        if (!streamMessage) {
                            removeLoadingIndicator(loaderId);
                            streamMessage = createStreamingBotMessage();
                        }
                        appendStreamingChunk(streamMessage, event.content || '');
                        continue;
                    }

                    if (event.type === 'final') {
                        if (!streamMessage) {
                            removeLoadingIndicator(loaderId);
                            streamMessage = createStreamingBotMessage();
                            if (event.answer) {
                                appendStreamingChunk(streamMessage, event.answer);
                            }
                        } else if (!streamMessage.rawText && event.answer) {
                            appendStreamingChunk(streamMessage, event.answer);
                        }
                        finalizeStreamingBotMessage(streamMessage, event);
                        continue;
                    }

                    if (event.type === 'error') {
                        throw new Error(event.message || 'The API request failed.');
                    }
                }
            }

            removeLoadingIndicator(loaderId);
            
            // Clear last attempt on success
            window.lastAttemptedQuery = "";

        } catch (error) {
            if (loaderId) removeLoadingIndicator(loaderId);
            // Render specific retry UI
            if(error.message === 'Failed to fetch') {
                 appendErrorMessage("网络请求失败，后端服务可能未启动或网络异常。", query);
            } else {
                 appendErrorMessage(error.message, query);
            }
        } finally {
            if (loaderId) removeLoadingIndicator(loaderId);
            setRequestLocked(false);
            userInput.focus();
        }
    };

    // Listeners
    sendBtn.addEventListener('click', () => handleSend());
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !isRequestLocked) handleSend();
    });
});
