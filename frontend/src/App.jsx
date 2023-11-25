import './App.css'
import Predictor from './components/Predictor'
import NFL from '../src/assets/NFL-logo.jpg'

function App() {
  return (
    <div className='all'>
      <div className="header">
        <img src={NFL} className='logo' alt="NFL Logo" />
        <h1>NFL Predictor</h1>
        <h3>Joshua Pfefferkorn & Jihwan Choi</h3>
      </div>
      <Predictor />
    </div>
  )
}

export default App
