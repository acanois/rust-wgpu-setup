const init = await import('./pkg/game.js');
init().then(() => console.log("WASM Loaded"));
