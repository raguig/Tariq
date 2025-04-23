import { io } from 'socket.io-client';

// Create the socket instance
export const socket = io('http://localhost:5000', {
  autoConnect: false, // We'll connect manually in the component
  reconnection: true,
  reconnectionAttempts: 5,
  reconnectionDelay: 1000
});

// Event logging for debugging
socket.on('connect', () => {
  console.log('Connected to socket server');
});

socket.on('connect_error', (error) => {
  console.error('Connection error:', error);
});

socket.on('reconnect_attempt', (attemptNumber) => {
  console.log(`Attempting to reconnect (${attemptNumber})`);
});

socket.on('reconnect_failed', () => {
  console.error('Failed to reconnect');
});

socket.on('disconnect', () => {
  console.log('Disconnected from socket server');
});

// Socket service functions
export const connectSocket = () => {
  if (!socket.connected) {
    socket.connect();
  }
};

export const disconnectSocket = () => {
  if (socket.connected) {
    socket.disconnect();
  }
};