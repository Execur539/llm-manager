import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
// The design system, in order: tokens, elements, the window, conversations, pages, components.
import './styles/tokens.css'
import './styles/base.css'
import './styles/shell.css'
import './styles/chat.css'
import './styles/views.css'
import './styles/components.css'

/*
 * What the page can rely on from the window it is in.
 *
 * `desktop`: the app's own title bar is the window's, so it is a drag region and leaves room for
 * the window buttons. `mica`: the window's base is the system material, so the body stays clear.
 */
const root = document.documentElement
if (window.api?.isDesktop) root.classList.add('desktop')
if (window.api?.mica) root.classList.add('mica')

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
