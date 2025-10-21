import { useState, useEffect } from 'react';
import { ShowOverlay, HideOverlay } from '../wailsjs/go/main/App';
import { EventsOn } from '../wailsjs/runtime/runtime';

function App() {
    const [query, setQuery] = useState('');
    
    useEffect(() => {
        // Listen for show-overlay event
        EventsOn('show-overlay', () => {
            ShowOverlay();
            // Focus input when overlay appears
            setTimeout(() => {
                document.getElementById('search')?.focus();
            }, 100);
        });

        // Handle Escape key to close
        const handleEscape = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                HideOverlay();
            }
        };
        
        window.addEventListener('keydown', handleEscape);
        return () => window.removeEventListener('keydown', handleEscape);
    }, []);
    
    return (
        <div className="overlay-container">
            <input
                id="search"
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search..."
            />
            {/* Your results/actions here */}
        </div>
    );
}

export default App;