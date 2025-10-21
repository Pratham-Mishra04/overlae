package main

import (
	"context"
	"runtime"
	"sync"

	wailsruntime "github.com/wailsapp/wails/v2/pkg/runtime"
	"golang.design/x/hotkey"
)

var (
	hotkeyInitOnce sync.Once
	hotkeyEvents   = make(chan struct{}, 1)
)

func init() {
	// Ensure we're on the main thread for initialization
	runtime.LockOSThread()
}

// App struct
type App struct {
	ctx context.Context
	hk  *hotkey.Hotkey
}

// NewApp creates a new App application struct
func NewApp() *App {
	return &App{}
}

// startup is called when the app starts. The context is saved
// so we can call the runtime methods
func (a *App) startup(ctx context.Context) {
	a.ctx = ctx

	hotkeyInitOnce.Do(func() {
		// Initialize hotkey on the main OS thread
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()

		// Register Cmd+G on macOS (ModCmd) or Ctrl+G on Windows/Linux
		hk := hotkey.New([]hotkey.Modifier{hotkey.ModCmd}, hotkey.KeyG)
		if err := hk.Register(); err != nil {
			return
		}
		a.hk = hk

		// Start hotkey listener in a separate goroutine
		go func() {
			for range hk.Keydown() {
				// Call ShowOverlay directly from the main thread
				wailsruntime.EventsEmit(ctx, "show-overlay")
			}
		}()

		// Start event handler
		go func() {
			for range hotkeyEvents {
				a.ShowOverlay()
			}
		}()
	})
}

func (a *App) ShowOverlay() {
	wailsruntime.WindowShow(a.ctx)
	wailsruntime.WindowCenter(a.ctx)
	wailsruntime.WindowSetAlwaysOnTop(a.ctx, true)
}

func (a *App) HideOverlay() {
	wailsruntime.WindowHide(a.ctx)
}

// Let users click outside to close
func (a *App) OnWindowBlur() {
	a.HideOverlay()
}

// Cleanup is called when the app is about to exit
func (a *App) Cleanup(ctx context.Context) bool {
	if a.hk != nil {
		_ = a.hk.Unregister()
	}
	return false // false means allow the app to close
}
