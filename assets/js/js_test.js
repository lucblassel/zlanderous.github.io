let sketch = function(p) {
    p.setup = function(){
      p.createCanvas(100, 100);
      p.background(0);
      console.log("instanced")
    }
  };
new p5(sketch, window.document.getElementById('myCanvas'));