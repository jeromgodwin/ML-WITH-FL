import React, { useEffect, useState, useRef } from 'react'

export default function ParallaxBackground() {
  const [mouse, setMouse] = useState({ x: 0, y: 0 })
  const rafRef = useRef(null)
  
  useEffect(() => {
    // Check for prefers-reduced-motion
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    if (mq.matches) return
    
    const handleMouseMove = (e) => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      rafRef.current = requestAnimationFrame(() => {
        // Normalize mouse coordinates to -1 to 1
        const x = (e.clientX / window.innerWidth) * 2 - 1
        const y = (e.clientY / window.innerHeight) * 2 - 1
        setMouse({ x, y })
      })
    }
    
    window.addEventListener('mousemove', handleMouseMove)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [])
  
  // Parallax offsets (subtle depth)
  const l1 = { transform: `translate(${mouse.x * 2}px, ${mouse.y * 2}px)` } // Surface
  const l2 = { transform: `translate(${mouse.x * -4}px, ${mouse.y * -4}px)` } // Tech Grid
  const l3 = { transform: `translate(${mouse.x * -8}px, ${mouse.y * -8}px)` } // Circuit Lines
  const l4 = { transform: `translate(${mouse.x * 12}px, ${mouse.y * 12}px)` } // Atmospheric depth
  const l5 = { transform: `translate(${mouse.x * -16}px, ${mouse.y * -16}px)` } // Particles
  
  return (
    <div className="fixed inset-0 z-[-1] overflow-hidden pointer-events-none bg-base-950">
      {/* Layer 1: Dark industrial surface */}
      <div 
        className="absolute -inset-[10%] bg-base-950 opacity-100" 
        style={l1} 
      />
      
      {/* Layer 2: Subtle technical grid */}
      <div 
        className="absolute -inset-[10%] opacity-20 bg-grid-tech bg-[length:40px_40px]" 
        style={l2} 
      />
      
      {/* Additional sub-grid for scale */}
      <div 
        className="absolute -inset-[10%] opacity-10 bg-grid-tech bg-[length:10px_10px]" 
        style={l2} 
      />
      
      {/* Layer 3: Thin circuit/technical lines (Simulated with a sparse repeating linear gradient) */}
      <div 
        className="absolute -inset-[10%] opacity-10" 
        style={{
          ...l3,
          backgroundImage: 'linear-gradient(90deg, transparent 99%, rgba(255,255,255,0.8) 100%), linear-gradient(0deg, transparent 99.5%, rgba(255,255,255,0.8) 100%)',
          backgroundSize: '200px 300px'
        }} 
      />
      
      {/* Layer 4: Soft atmospheric depth (low contrast glowing orbs in corners) */}
      <div className="absolute -inset-[10%]" style={l4}>
        <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-console-blue/5 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-console-cyan/5 rounded-full blur-[120px]" />
      </div>
      
      {/* Layer 5: Extremely subtle particles (Simulated with noise/dots) */}
      <div 
        className="absolute -inset-[10%] opacity-[0.015]" 
        style={{
          ...l5,
          backgroundImage: 'url("data:image/svg+xml,%3Csvg viewBox=%220 0 200 200%22 xmlns=%22http://www.w3.org/2000/svg%22%3E%3Cfilter id=%22noiseFilter%22%3E%3CfeTurbulence type=%22fractalNoise%22 baseFrequency=%220.85%22 numOctaves=%223%22 stitchTiles=%22stitch%22/%3E%3C/filter%3E%3Crect width=%22100%25%22 height=%22100%25%22 filter=%22url(%23noiseFilter)%22/%3E%3C/svg%3E")',
          mixBlendMode: 'screen'
        }} 
      />
      
      {/* Vignette mask for depth */}
      <div className="absolute inset-0 bg-gradient-to-t from-base-950/80 via-transparent to-base-950/40" />
    </div>
  )
}
