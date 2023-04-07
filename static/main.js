var socket = io('http://localhost:5000');

socket.on('connect', function(){
    console.log("Connected...!", socket.connected)
});

const video = document.querySelector("#videoElement");

video.width = 400; 
video.height = 300; ;

if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
    .then(function (stream) {
        video.srcObject = stream;
        video.play();
    })
    .catch(function (err0r) {
        console.log(err0r)
        console.log("Something went wrong!");
    });
}

function capture(video, scaleFactor) {
    if(scaleFactor == null){
        scaleFactor = 1;
    }
    var w = video.videoWidth * scaleFactor;
    var h = video.videoHeight * scaleFactor;
    var canvas = document.createElement('canvas');
        canvas.width  = w;
        canvas.height = h;
    var ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, w, h);
    return canvas;
}

const FPS = 22;

setInterval(() => {
    // cap.read(src);

    var type = "image/png"
    var video_element = document.getElementById("videoElement")
    var frame = capture(video_element, 1)
    var data = frame.toDataURL(type);

    data = data.replace('data:' + type + ';base64,', '');
    
    socket.emit('image', data);
}, 10000/FPS);


socket.on('response_back', function(image){
    const image_id = document.getElementById('image');
    image_id.src = image;
});

socket.on('class_response', function(class_name){
    const class_paragraph = document.getElementById('inferred_class');
    class_paragraph.textContent = class_name
});