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
          <title> Flower Classifier </title>
          <meta name="Flower Classifier" content="Different models that determines the type of flower." />
      </Helmet>
      <NavBar />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Model />} />
          <Route path="/model" element={<Model />} />
          <Route path="/info" element={<Info />} />
        </Routes>
      </BrowserRouter>
      <FooterBar/>

    </ChakraProvider>
  );
}

export default App;
