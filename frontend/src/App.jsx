import { BrowserRouter, Routes, Route } from "react-router-dom";
import Prediction from "./pages/Prediction";
import "./App.css";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Prediction />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
