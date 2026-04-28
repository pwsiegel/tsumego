import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './theme.css'
import './index.css'
import App from './App.tsx'

// Apply persisted theme choice before first render to avoid a flash.
const savedTheme = localStorage.getItem('theme')
if (savedTheme === 'dark') {
  document.documentElement.dataset.theme = 'dark'
} else if (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches) {
  document.documentElement.dataset.theme = 'dark'
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
)
