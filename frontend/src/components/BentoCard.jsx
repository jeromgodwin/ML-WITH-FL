import React from 'react'
import { motion } from 'framer-motion'

/**
 * Grid container that orchestrates the cascading stagger for its BentoCard children.
 * Wrap a set of <BentoCard /> elements in <BentoGrid> so they animate in as a sequence
 * on view/tab mount, instead of each card animating independently.
 *
 * @param {object} props
 * @param {React.ReactNode} props.children
 * @param {string} [props.className] - grid column/gap utilities, e.g. "grid-cols-4 auto-rows-[140px]"
 */
export function BentoGrid({ children, className = '' }) {
  return (
    <motion.div
      className={`grid gap-4 ${className}`}
      variants={gridVariants}
      initial="hidden"
      animate="show"
    >
      {children}
    </motion.div>
  )
}

const gridVariants = {
  hidden: {},
  show: {
    transition: {
      staggerChildren: 0.06,
      delayChildren: 0.05,
    },
  },
}

const cardVariants = {
  hidden: { opacity: 0, y: 16, scale: 0.98 },
  show: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] },
  },
}

const GLOW = {
  cyan: {
    hoverShadow:
      '0 0 0 1px rgba(34,211,238,0.35), 0 0 40px -6px rgba(34,211,238,0.55)',
    ring: 'group-hover:ring-cyan-glow/40',
  },
  violet: {
    hoverShadow:
      '0 0 0 1px rgba(168,85,247,0.35), 0 0 40px -6px rgba(168,85,247,0.55)',
    ring: 'group-hover:ring-violet-glow/40',
  },
  neutral: {
    hoverShadow:
      '0 0 0 1px rgba(255,255,255,0.14), 0 0 32px -8px rgba(255,255,255,0.18)',
    ring: 'group-hover:ring-white/20',
  },
}

/**
 * BentoCard — the reusable glass tile for the dashboard grid.
 *
 * Must be rendered inside a <BentoGrid> to get the cascading entrance stagger;
 * used standalone it still animates in on mount, just without sequencing against siblings.
 *
 * @param {object} props
 * @param {React.ReactNode} [props.title] - card heading, rendered in the header row
 * @param {React.ReactNode} [props.icon] - optional leading icon element (e.g. lucide-react icon)
 * @param {React.ReactNode} [props.eyebrow] - small label above the title (e.g. "ROUND 42")
 * @param {React.ReactNode} props.children - card body content
 * @param {'cyan'|'violet'|'neutral'} [props.glow='cyan'] - hover glow / accent color
 * @param {string} [props.span] - Tailwind col/row span utilities, e.g. "col-span-2 row-span-2"
 * @param {boolean} [props.interactive=true] - disable hover lift/glow for static/read-only tiles
 * @param {() => void} [props.onClick] - optional click handler, makes the card a button
 * @param {string} [props.className] - extra utility classes for the inner content wrapper
 */
export default function BentoCard({
  title,
  icon,
  eyebrow,
  children,
  glow = 'cyan',
  span = '',
  interactive = true,
  onClick,
  className = '',
}) {
  const { hoverShadow, ring } = GLOW[glow] ?? GLOW.cyan
  const Tag = onClick ? motion.button : motion.div

  return (
    <Tag
      variants={cardVariants}
      whileHover={
        interactive
          ? { scale: 1.02, boxShadow: hoverShadow, transition: { duration: 0.2, ease: 'easeOut' } }
          : undefined
      }
      onClick={onClick}
      className={[
        'group relative overflow-hidden rounded-2xl',
        'bg-white/5 backdrop-blur-md',
        'border border-white/10',
        'shadow-glass',
        'ring-1 ring-inset ring-transparent transition-[box-shadow] duration-200',
        interactive ? ring : '',
        onClick ? 'text-left cursor-pointer' : '',
        span,
      ].join(' ')}
    >
      {/* faint top-edge highlight to sell the glass edge */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/25 to-transparent" />

      <div className={`relative flex h-full flex-col p-5 ${className}`}>
        {(title || icon || eyebrow) && (
          <div className="mb-3 flex items-start justify-between">
            <div>
              {eyebrow && (
                <div className="mb-1 font-mono text-[10px] uppercase tracking-wider text-zinc-500">
                  {eyebrow}
                </div>
              )}
              {title && (
                <h3 className="font-display text-sm font-medium text-zinc-200">{title}</h3>
              )}
            </div>
            {icon && (
              <div className="text-zinc-500 group-hover:text-zinc-300 transition-colors">
                {icon}
              </div>
            )}
          </div>
        )}

        <div className="flex-1 min-h-0">{children}</div>
      </div>
    </Tag>
  )
}
