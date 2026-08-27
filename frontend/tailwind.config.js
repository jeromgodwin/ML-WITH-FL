/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        base: {
          950: '#0a0a0a',
          900: '#111113',
          800: '#1a1a1e',
        },
        cyan: {
          glow: '#22d3ee',
        },
        violet: {
          glow: '#a855f7',
        },
      },
      fontFamily: {
        display: ['"Space Grotesk"', 'sans-serif'],
        sans: ['"Inter"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      boxShadow: {
        'glow-cyan': '0 0 0 1px rgba(34,211,238,0.15), 0 0 24px -4px rgba(34,211,238,0.35)',
        'glow-violet': '0 0 0 1px rgba(168,85,247,0.15), 0 0 24px -4px rgba(168,85,247,0.35)',
        'glow-cyan-lg': '0 0 60px -10px rgba(34,211,238,0.5)',
        'glow-violet-lg': '0 0 60px -10px rgba(168,85,247,0.5)',
        glass: 'inset 0 1px 0 0 rgba(255,255,255,0.06), 0 8px 32px -8px rgba(0,0,0,0.6)',
      },
      backgroundImage: {
        'grid-faint':
          'linear-gradient(to right, rgba(255,255,255,0.035) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.035) 1px, transparent 1px)',
      },
      backgroundSize: {
        grid: '32px 32px',
      },
      animation: {
        'blob-a': 'blobMove 22s ease-in-out infinite',
        'blob-b': 'blobMove 30s ease-in-out infinite reverse',
      },
      keyframes: {
        blobMove: {
          '0%, 100%': { transform: 'translate(0,0) scale(1)' },
          '33%': { transform: 'translate(4%, -6%) scale(1.08)' },
          '66%': { transform: 'translate(-3%, 4%) scale(0.96)' },
        },
      },
    },
  },
  plugins: [
    function ({ addUtilities }) {
      addUtilities({
        '.glass': {
          background: 'rgba(255,255,255,0.04)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          border: '1px solid rgba(255,255,255,0.08)',
        },
        '.glass-raised': {
          background: 'rgba(255,255,255,0.06)',
          backdropFilter: 'blur(24px)',
          WebkitBackdropFilter: 'blur(24px)',
          border: '1px solid rgba(255,255,255,0.10)',
        },
      })
    },
  ],
}
