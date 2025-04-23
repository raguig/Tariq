import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getAnalytics } from "firebase/analytics";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBZMBdxLSvNfI_xEth5_BiK5A3MRCmSO0E",
  authDomain: "tariq-a1502.firebaseapp.com",
  projectId: "tariq-a1502",
  storageBucket: "tariq-a1502.appspot.com", // Updated to standard format
  messagingSenderId: "445125399201",
  appId: "1:445125399201:web:85f19c97aae6369e43d6e0",
  measurementId: "G-3KLFR40KSQ"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
const db = getFirestore(app);

export { db, analytics };