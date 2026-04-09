// Global Navigation
document.addEventListener('DOMContentLoaded', function() {
    initSidebar();
    initTextQuiz();
    initVideoAnalysis();
    initResultsAnimations();
});

// Sidebar functionality
function initSidebar() {
    const sidebar = document.getElementById('sidebar');
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const overlay = document.getElementById('overlay');

    if (mobileMenuBtn) {
        mobileMenuBtn.addEventListener('click', () => {
            sidebar?.classList.remove('-translate-x-full');
            overlay?.classList.remove('hidden');
        });
    }

    if (overlay) {
        overlay.addEventListener('click', () => {
            sidebar?.classList.add('-translate-x-full');
            overlay.classList.add('hidden');
        });
    }
}

// Text Quiz Functionality
function initTextQuiz() {
    const questions = [
        { title: "How have you been feeling lately?", desc: "Describe your emotions and experiences." },
        { title: "What activities bring you joy?", desc: "Things that make you happy or fulfilled." },
        { title: "How do you handle stress?", desc: "Your coping mechanisms when overwhelmed." },
        { title: "How is your sleep and energy?", desc: "Sleep patterns and daily energy levels." },
        { title: "What are your social connections like?", desc: "Relationships and support network." }
    ];

    let currentQuestion = 0;
    let answers = [];

    const elements = {
        container: document.getElementById('quizContainer'),
        currentQuestion: document.getElementById('currentQuestion'),
        progressBar: document.getElementById('progressBar'),
        progressText: document.getElementById('progressText'),
        answerInput: document.getElementById('answerInput'),
        nextBtn: document.getElementById('nextBtn'),
        prevBtn: document.getElementById('prevBtn'),
        questionTitle: document.getElementById('questionTitle'),
        questionDesc: document.getElementById('questionDesc')
    };

    if (!elements.container) return;

    function updateQuestion() {
        const q = questions[currentQuestion];
        elements.questionTitle.textContent = q.title;
        elements.questionDesc.textContent = q.desc;
        elements.currentQuestion.textContent = currentQuestion + 1;
        
        const progress = ((currentQuestion + 1) / questions.length) * 100;
        elements.progressBar.style.width = progress + '%';
        elements.progressText.textContent = Math.round(progress) + '%';

        elements.prevBtn.style.display = currentQuestion === 0 ? 'none' : 'flex';
        elements.nextBtn.innerHTML = currentQuestion === questions.length - 1 
            ? '<i class="fas fa-flag-checkered mr-2"></i> Get Results'
            : '<i class="fas fa-arrow-right mr-2"></i> Next';

        elements.answerInput.focus();
    }

    elements.nextBtn.addEventListener('click', () => {
        const answer = elements.answerInput.value.trim();
        if (answer || currentQuestion === questions.length - 1) {
            answers[currentQuestion] = answer;
            
            if (currentQuestion < questions.length - 1) {
                currentQuestion++;
                elements.answerInput.value = answers[currentQuestion] || '';
                updateQuestion();
            } else {
                // Simulate API call
                elements.nextBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Analyzing...';
                elements.nextBtn.disabled = true;
                setTimeout(() => {
                    localStorage.setItem('assessmentType', 'text');
                    window.location.href = 'results.html';
                }, 2000);
            }
        } else {
            alert('Please share your thoughts before continuing.');
        }
    });

    elements.prevBtn.addEventListener('click', () => {
        if (currentQuestion > 0) {
            currentQuestion--;
            elements.answerInput.value = answers[currentQuestion] || '';
            updateQuestion();
        }
    });

    elements.answerInput.addEventListener('input', () => {
        answers[currentQuestion] = elements.answerInput.value;
    });

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            elements.nextBtn.click();
        }
    });

    updateQuestion();
}

// Video Analysis Functionality
function initVideoAnalysis() {
    const video = document.getElementById('videoPlayer');
    const controlsOverlay = document.getElementById('controlsOverlay');
    const recordingIndicator = document.getElementById('recordingIndicator');
    const startRecordingBtn = document.getElementById('startRecording');
    const questionDisplay = document.getElementById('questionDisplay');
    const currentQuestionEl = document.getElementById('currentQuestion');
    const timerEl = document.getElementById('timer');
    const switchToTextBtn = document.getElementById('switchToText');

    let mediaRecorder, stream, recordedChunks = [];
    let currentVideoQuestion = 0;
    let recordingTimer;
    let timeLeft = 30;

    const videoQuestions = [
        "Tell us how you've been feeling this week.",
        "What brings you the most joy in life?",
        "How do you typically handle difficult emotions?"
    ];

    if (!video || !startRecordingBtn) return;

    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480, facingMode: 'user' },
                audio: true 
            });
            video.srcObject = stream;
            
            controlsOverlay.style.opacity = '0';
            startRecordingBtn.style.display = 'none';
            questionDisplay.classList.remove('hidden');
            
            setTimeout(() => startRecording(), 1000);
        } catch (err) {
            alert('Camera access denied. Please allow camera and microphone permissions.');
        }
    }

    function startRecording() {
        const chunks = [];
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
        mediaRecorder.onstop = () => {
            recordedChunks = chunks;
            recordingIndicator.classList.add('opacity-0');
            if (currentVideoQuestion < videoQuestions.length - 1) {
                setTimeout(() => nextVideoQuestion(), 1500);
            } else {
                // Simulate analysis
                setTimeout(() => {
                    localStorage.setItem('assessmentType', 'video');
                    window.location.href = 'results.html';
                }, 2000);
            }
        };

        mediaRecorder.start();
        recordingIndicator.classList.remove('opacity-0');
        showVideoQuestion();
    }

    function showVideoQuestion() {
        currentQuestionEl.textContent = videoQuestions[currentVideoQuestion];
        timeLeft = 30;
        timerEl.textContent = timeLeft;
        recordingTimer = setInterval(() => {
            timeLeft--;
            timerEl.textContent = timeLeft;
            if (timeLeft <= 0) {
                clearInterval(recordingTimer);
                mediaRecorder.stop();
            }
        }, 1000);
    }

    function nextVideoQuestion() {
        currentVideoQuestion++;
        startRecording();
    }

    startRecordingBtn.addEventListener('click', startCamera);
    switchToTextBtn.addEventListener('click', () => {
        if (stream) stream.getTracks().forEach(track => track.stop());
        window.location.href = 'text-quiz.html';
    });

    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        if (stream) stream.getTracks().forEach(track => track.stop());
    });
}

// Results Page Animations
function initResultsAnimations() {
    // Animate score counter
    const scoreEl = document.querySelector('.text-7xl');
    if (scoreEl) {
        let count = 0;
        const target = 72;
        const increment = target / 100;
        const timer = setInterval(() => {
            count += increment;
            scoreEl.textContent = Math.floor(count);
            if (count >= target) {
                scoreEl.textContent = target;
                clearInterval(timer);
            }
        }, 30);
    }

    // Fade in elements
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
            }
        });
    }, observerOptions);

    document.querySelectorAll('.animate-on-scroll').forEach(el => {
        observer.observe(el);
    });
}

// Login Form
const loginForm = document.getElementById('loginForm');
if (loginForm) {
    loginForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const submitBtn = loginForm.querySelector('button[type="submit"]');
        const originalText = submitBtn.innerHTML;
        
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Signing In...';
        submitBtn.disabled = true;
        
        setTimeout(() => {
            window.location.href = 'main.html';
        }, 1500);
    });
}

// Add animate-on-scroll class to elements that should animate
document.querySelectorAll('.bg-white, .bg-gradient-to-br, .grid').forEach(el => {
    el.classList.add('animate-on-scroll', 'opacity-0');
});

// Page transitions
window.addEventListener('pageshow', (event) => {
    if (event.persisted) {
        window.location.reload();
    }
});

// Service Worker for PWA (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js');
    });
}