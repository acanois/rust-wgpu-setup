const init = await import('./pkg/wgpu_template.js');
init().then(() => console.log("WASM Loaded"));
