import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { ThemeProvider } from 'react-hook-theme';
import { HashRouter } from 'react-router-dom';
import { AlertContainer } from '@/components/Alerts/AlertContainer.tsx';
import { AlertsProvider } from '@/components/Alerts/context/AlertsContext';
import App from './App';
import './styles/global.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <HashRouter>
      <ThemeProvider options={{ theme: 'dark', save: true }}>
        <AlertsProvider>
          <App />
          <AlertContainer />
        </AlertsProvider>
      </ThemeProvider>
    </HashRouter>
  </StrictMode>,
);
