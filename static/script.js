var socket = io();
let room, username;
let localStream, remoteStream;
let peerConnection;

// Chatbot Functionality
async function askChatbot() {
  let userMessage = document.getElementById("chatbot-input").value;
  if (!userMessage) return; // Avoid empty requests

  try {
    let response = await fetch("/chatbot/query", {
      method: "POST",
      body: JSON.stringify({ message: userMessage }),
      headers: { "Content-Type": "application/json" },
    });

    if (!response.ok) {
      throw new Error("Chatbot request failed!");
    }

    let result = await response.json();
    document.getElementById("chatbot-response").innerText = result.response;
  } catch (error) {
    console.error("Error:", error);
    document.getElementById("chatbot-response").innerText =
      "Error contacting chatbot!";
  }
}

// Video Chat Functionality
function joinRoom() {
  room = document.getElementById("room").value;
  username = document.getElementById("username").value;
  socket.emit("join", { room: room, username: username });
}

socket.on("user_joined", (data) => {
  let chat = document.getElementById("chat");
  chat.innerHTML += <b>${data.username} joined the room</b><br>;
});

function sendMessage() {
  let message = document.getElementById("message").value;
  socket.emit("message", { room: room, username: username, message: message });
}

socket.on("message", (data) => {
  let chat = document.getElementById("chat");
  if (data.username !== username) {
    // Don't display your own message
    chat.innerHTML += <b>${data.username}:</b> ${data.message}<br>;
  }
});

// WebRTC Video Call Setup
async function startCall() {
  localStream = await navigator.mediaDevices.getUserMedia({
    video: true,
    audio: true,
  });
  document.getElementById("localVideo").srcObject = localStream;

  peerConnection = new RTCPeerConnection();
  localStream
    .getTracks()
    .forEach((track) => peerConnection.addTrack(track, localStream));

  peerConnection.onicecandidate = (event) => {
    if (event.candidate) {
      socket.emit("ice-candidate", { room: room, candidate: event.candidate });
    }
  };

  peerConnection.ontrack = (event) => {
    document.getElementById("remoteVideo").srcObject = event.streams[0];
  };

  const offer = await peerConnection.createOffer();
  await peerConnection.setLocalDescription(offer);
  socket.emit("offer", { room: room, offer: offer });
}

socket.on("offer", async (data) => {
  peerConnection = new RTCPeerConnection();
  remoteStream = new MediaStream();
  document.getElementById("remoteVideo").srcObject = remoteStream;

  peerConnection.ontrack = (event) => {
    remoteStream.addTrack(event.track);
  };

  peerConnection.onicecandidate = (event) => {
    if (event.candidate) {
      socket.emit("ice-candidate", { room: room, candidate: event.candidate });
    }
  };

  await peerConnection.setRemoteDescription(
    new RTCSessionDescription(data.offer)
  );
  const answer = await peerConnection.createAnswer();
  await peerConnection.setLocalDescription(answer);
  socket.emit("answer", { room: room, answer: answer });
});

socket.on("answer", async (data) => {
  await peerConnection.setRemoteDescription(
    new RTCSessionDescription(data.answer)
  );
});

socket.on("ice-candidate", async (data) => {
  if (data.candidate) {
    await peerConnection.addIceCandidate(new RTCIceCandidate(data.candidate));
  }
});
