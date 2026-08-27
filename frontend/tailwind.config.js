/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        base: {
          950: '#0b0d0e',
          900: '#111315',
          800: '#171a1c',
          700: '#1f2326',
        },
        console: {
          green: '#10b981',
          red: '#ef4444',
          orange: '#f59e0b',
          yellow: '#eab308',
          blue: '#3b82f6',
          cyan: '#06b6d4',
          gray: '#52525b'
        }
      },
      fontFamily: {
        display: ['"Space Grotesk"', 'sans-serif'],
        sans: ['"Inter"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      boxShadow: {
        'panel-raised': '0 2px 0 0 rgba(255, 255, 255, 0.05) inset, 0 -1px 0 0 rgba(0,0,0,0.4) inset, 0 4px 8px -2px rgba(0, 0, 0, 0.8), 0 0 1px 1px rgba(0,0,0,0.9)',
        'panel-recessed': 'inset 0 4px 8px -2px rgba(0, 0, 0, 0.9), inset 0 2px 4px rgba(0,0,0,0.8), inset 0 -1px 0 0 rgba(255,255,255,0.04), 0 1px 0 0 rgba(255,255,255,0.06)',
        'btn-raised': 'inset 0 1px 0 0 rgba(255, 255, 255, 0.15), inset 0 -1px 0 0 rgba(0,0,0,0.6), 0 2px 4px -1px rgba(0,0,0,0.8), 0 1px 1px rgba(0,0,0,0.9)',
        'btn-pressed': 'inset 0 2px 4px rgba(0, 0, 0, 0.6), inset 0 1px 2px rgba(0,0,0,0.9), 0 1px 0 0 rgba(255,255,255,0.05)',
        'led-green': '0 0 8px 1px rgba(16, 185, 129, 0.6)',
        'led-red': '0 0 8px 1px rgba(239, 68, 68, 0.6)',
        'led-orange': '0 0 8px 1px rgba(245, 158, 11, 0.6)',
        'led-blue': '0 0 8px 1px rgba(59, 130, 246, 0.6)',
        'led-cyan': '0 0 8px 1px rgba(6, 182, 212, 0.6)',
      },
      backgroundImage: {
        'grid-tech': 'linear-gradient(to right, rgba(255,255,255,0.02) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.02) 1px, transparent 1px)',
        'metal-texture': 'linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0) 100%)',
      },
      backgroundSize: {
        grid: '40px 40px',
        gridSmall: '10px 10px',
      }
    },
  },
  plugins: [],
}
