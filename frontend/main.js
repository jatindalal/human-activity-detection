const server_address = "localhost"
const port = "80"

const video = document.querySelector("#videoElement");

video.width = 400;
video.height = 300;

const FPS = 10;

let inferenceIntervalId = null;

if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
    .then(function (stream) {
        video.srcObject = stream;
        video.play();
    })
    .catch(function (err0r) {
        console.log(err0r)
        console.log("Can't access Webcam");
    });
}

function capture(video, scaleFactor) {
    if (scaleFactor == null) {
        scaleFactor = 1;
    }

    var w = video.videoWidth * scaleFactor;
    var h = video.videoHeight * scaleFactor;

    var canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;

    var ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, w, h);
    return canvas;
}

let updateImageAndClass = () => {
    var type = "image/png"
    var video_element = document.getElementById("videoElement")
    var frame = capture(video_element, 1)
    var data = frame.toDataURL(type);

    data = data.replace(`data:${type};base64,`, '');

    fetch("http://" + server_address + ":" + port + "/uploadimage/", {
        body: "filedata="+encodeURIComponent(data),
        headers: {
            Accept: "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
        },
        method: "POST"
    })
    .then((response) => response.json())
    .then((json) => {
        document.getElementById("image").src = json.masked_image
        document.getElementById("inferred_class").innerText = "Activity Detected: " + json.class
    });

}

const toggleButton = document.getElementById("toggle_inference");
toggleButton.addEventListener("click", () => {
    // console.log("clicked!")
    if (inferenceIntervalId == null) {
        inferenceIntervalId = setInterval(updateImageAndClass, 10000 / FPS);

        toggleButton.innerText = "Stop Inference";
    }
    else {
        clearInterval(inferenceIntervalId);
        inferenceIntervalId = null;

        toggleButton.innerText = "Start Inference";
    }

});








