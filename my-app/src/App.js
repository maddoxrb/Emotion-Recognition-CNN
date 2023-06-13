import {
  React,
} from 'react'

import { 
  ChakraProvider,
} from '@chakra-ui/react'

import {Helmet} from "react-helmet";

import { 
  BrowserRouter,
  Routes,
  Route,
 } from "react-router-dom";


// Import Pages for the Main Page
import NavBar from './NavBar';
import FooterBar from './FooterBar';
import Info from './Info';
import Model from './Model';


function App() {

  return (
    <ChakraProvider>
      <Helmet>
          <title> Face Emotion Detector </title>
          <meta name="Face Emotion Detector" content="A CNN model that determines the emotion of faces." />
      </Helmet>
      <NavBar />
      <BrowserRouter>
        <Routes>
          <Route path="" element={<Model />} />
          <Route path="model" element={<Model />} />
          <Route path="info" element={<Info />} />
        </Routes>
      </BrowserRouter>
      <FooterBar/>

    </ChakraProvider>
  );
}

export default App;
