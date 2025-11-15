/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // 🔵 PRIMARY COLORS (Blue Theme)
        primary: {
          DEFAULT: '#3B82F6',  // Blue 500 - Main brand color
          hover: '#2563EB',     // Blue 600 - Hover state
          light: '#60A5FA',     // Blue 400 - Lighter variant
          dark: '#1D4ED8',      // Blue 700 - Darker variant
        },

        // 🌑 BACKGROUND COLORS (Dark Theme)
        background: {
          primary: '#0F172A',   // Slate 900 - Main background
          secondary: '#1E293B', // Slate 800 - Secondary surfaces
          card: '#1E293B',      // Slate 800 - Card background
        },

        // 📝 TEXT COLORS
        text: {
          primary: '#F1F5F9',   // Slate 100 - Main text
          secondary: '#94A3B8', // Slate 400 - Secondary text
          muted: '#64748B',     // Slate 500 - Muted text
        },

        // 🔲 BORDER COLORS
        border: {
          DEFAULT: '#334155',   // Slate 700 - Default borders
          light: '#475569',     // Slate 600 - Lighter borders
          dark: '#1E293B',      // Slate 800 - Darker borders
        },

        // ✅ STATUS COLORS (Keep existing - they're perfect)
        status: {
          success: '#10B981',   // Green 500 - Compliant
          warning: '#F59E0B',   // Amber 500 - Warning
          danger: '#EF4444',    // Red 500 - Critical
          info: '#3B82F6',      // Blue 500 - Info
        },

        // 🎨 ACCENT COLORS (for charts, highlights, etc.)
        accent: {
          purple: '#A855F7',    // Purple 500 - Keep for mitigation badges
          teal: '#14B8A6',      // Teal 500 - Alternative accent
          cyan: '#06B6D4',      // Cyan 500 - Charts
          indigo: '#6366F1',    // Indigo 500 - Charts
        }
      },

      // 📊 BOX SHADOW
      boxShadow: {
        'card': '0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)',
        'card-hover': '0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -2px rgba(0, 0, 0, 0.3)',
      },

      // 🎯 BORDER RADIUS
      borderRadius: {
        'card': '0.75rem',  // 12px
      }
    },
  },
  plugins: [],
}