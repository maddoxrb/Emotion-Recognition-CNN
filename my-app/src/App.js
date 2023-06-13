import {
  React,
  useState,
  useRef,
} from 'react'

import AccuracyChart from './AccuracyChart';
import InstructionsCard from './InstructionsCard';
import NavBar from './NavBar';
import FooterBar from './FooterBar';
import Info from './Info';

import { 
  TabList, 
  TabPanels, 
  Tab, 
  Tabs,
  TabPanel, 
  ChakraProvider,
  Center,
  Heading,
  Box,
  VStack,
  Image,
  Card,
  CardBody,
  CardHeader,
  SimpleGrid,
  Divider,
  CardFooter,
  Text,
  AbsoluteCenter,
  Button,
  HStack,
  StackDivider,
  Flex,
  Alert, 
  AlertIcon, 
  AlertDialog, 
  AlertDialogOverlay, 
  AlertDialogContent, 
  AlertDialogHeader, 
  AlertDialogCloseButton, 
  AlertDialogBody, 
  AlertDialogFooter,
  useDisclosure,
} from '@chakra-ui/react'

import {Helmet} from "react-helmet";

import { 
  BrowserRouter,
  Routes,
  Route,
 } from "react-router-dom";

const generatedImages = [
  { title: 'Generated Image', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'model'},
  // { title: 'Image 2', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'actual'},
  // { title: 'Image 3', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'actual'},
  // { title: 'Image 4', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'actual'},
  // { title: 'Image 5', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'actual'},
  // { title: 'Image 6', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'actual'},
  // { title: 'Image 7', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'actual'},
  // { title: 'Image 8', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'actual'},
  // { title: 'Image 9', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'actual'},
  // { title: 'Image 10', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'actual'},
];

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
          <Route path="" element={<FooterBar />} />
          <Route path="model" element={<FooterBar />} />
          <Route path="info" element={<Info />} />
        </Routes>
      </BrowserRouter>
      <FooterBar/>

    </ChakraProvider>
  );
}

export default App;
