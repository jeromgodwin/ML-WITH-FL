import React from 'react'
import { motion } from 'framer-motion'

export function BentoGrid({ children, className = '' }) {
  return (
    <motion.div
      className={`grid gap-5 ${className}`}
      variants={{
        hidden: {},
        show: { transition: { staggerChildren: 0.05, delayChildren: 0.02 } },
      }}
      initial="hidden"
      animate="show"
    >
      {children}
    </motion.div>
  )
}

export default function BentoCard({
  title,
  icon,
  eyebrow,
  children,
  glow = 'neutral',
  span = '',
  interactive = false,
  onClick,
  className = '',
  recessed = false,
}) {
  const Tag = onClick ? motion.button : motion.div

  // For skeuomorphism, we apply panel-raised or panel-recessed
  const panelClass = recessed ? 'panel-recessed' : 'panel-raised'
  
  return (
    <Tag
      variants={{
        hidden: { opacity: 0, y: 8 },
        show: { opacity: 1, y: 0, transition: { duration: 0.3, ease: 'easeOut' } },
      }}
      whileHover={interactive ? { scale: 1.01, zIndex: 10 } : undefined}
      whileTap={onClick ? { scale: 0.99, translateY: 1 } : undefined}
      onClick={onClick}
      className={`
        relative flex flex-col p-5
        ${panelClass}
        ${onClick ? 'cursor-pointer text-left' : ''}
        ${span}
      `}
    >
      {/* LED indicator line for top edge to simulate hardware */}
      {!recessed && <div className="absolute top-0 inset-x-4 h-[1px] bg-white/5" />}

      {(title || icon || eyebrow) && (
        <div className="mb-4 flex items-start justify-between border-b border-black/30 pb-3 shadow-[0_1px_0_rgba(255,255,255,0.02)]">
          <div>
            {eyebrow && (
              <div className="mb-1 font-mono text-[10px] uppercase tracking-widest text-zinc-500 font-semibold">
                {eyebrow}
              </div>
            )}
            {title && (
              <h3 className="font-display text-sm font-bold text-zinc-200 tracking-wide flex items-center gap-2">
                {/* Optional glow indicator based on type */}
                {glow === 'cyan' && <div className="led led-cyan" />}
                {glow === 'violet' && <div className="led led-blue" />}
                {glow === 'red' && <div className="led led-red" />}
                {glow === 'green' && <div className="led led-green" />}
                {glow === 'orange' && <div className="led led-orange" />}
                {title}
              </h3>
            )}
          </div>
          {icon && <div className="text-zinc-500">{icon}</div>}
        </div>
      )}

      <div className={`flex-1 min-h-0 ${className}`}>{children}</div>
    </Tag>
  )
}
