import * as wasm from "wasm-wgsl-canvas";


let btn = document.getElementById('submit');
console.log(btn);
btn.addEventListener('click', function() {
    var textarea = document.getElementById('textarea-1');
    var text = textarea.value;
    console.log(text);

    var canvas = document.getElementById('canvas-1');
    canvas.setAttribute('data-frag', text);
});


wasm.run();
