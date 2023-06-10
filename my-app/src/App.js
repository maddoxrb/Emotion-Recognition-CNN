import {
  React,
  useState,
  useRef,
} from 'react'

import AccuracyChart from './AccuracyChart';
import InstructionsCard from './InstructionsCard';

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

  // Button Feature for generating new image
  const [isButtonClicked, setIsButtonClicked] = useState(false);
  const cancelRef = useRef();
  const { isOpen, onOpen, onClose } = useDisclosure();

  // Button for determing what the user deems to be the emotion of the user
  const [selectedLabel, setSelectedLabel] = useState('');

  // Defines the states for altert message, data correctness, and selected label
  const [alertMessage, setAlertMessage] = useState('');
  const [dataCorrect, setDataCorrect] = useState(0);
  const [dataIncorrect, setDataIncorrect] = useState(0);

  const data = [
    { name: 'Correct', value: dataCorrect },
    { name: 'Incorrect', value: dataIncorrect },
  ];



  const handleClick = async () => {
    const runScript = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/run-script', {
          method: 'POST',
        });
        const data = await response.json();
        console.log(data);
        return data.result;
      } catch (error) {
        console.error('Error:', error);
        throw error;
      }
    };

    setIsButtonClicked(true);

  
    if (selectedLabel === '') {
      setAlertMessage('Please select an emotion classification before confirming and generating a new image.');
      onOpen(); // Move onOpen() inside else block
    } else if (selectedLabel === 'angry') {
      const result = await runScript();
      setDataCorrect(dataCorrect + parseInt(result));
      setSelectedLabel('');
    } else {
      setDataIncorrect(dataIncorrect + 1);
      setSelectedLabel('');
      setAlertMessage('');
    }
  };
  


  return (
    <ChakraProvider>
      <Helmet>
          <title> Face Emotion Detector </title>
          <meta name="Face Emotion Detector" content="A CNN model that determines the emotion of faces." />
      </Helmet>
        <Tabs isFitted variant='line' size='sm'>
          <TabList bg="blue.100">
            <Tab>Face Emotion Detector</Tab>
            <Tab >Info</Tab>
          </TabList>

          <TabPanels>
            <TabPanel bg="gray.200">
              <Center>
                <Heading pt={5} pb={0} as='h2' size='4xl'>
                  Face Emotion Detector
                </Heading>
              </Center>
              <InstructionsCard />
              <Flex height="75vh">
                <Flex width="30%" justifyContent="center" alignItems="center">
                  <Card height="100%" width="90%">
                    <Center>
                        <CardHeader>
                          <Heading size='lg'>Accuracy</Heading>
                        </CardHeader>
                      </Center>
                    <AbsoluteCenter>
                      <AccuracyChart data={data} />
                    </AbsoluteCenter>
                  </Card>
                </Flex>
                <Flex width="40%" justifyContent="center" alignItems="center">
                  {/* <SimpleGrid m={10} spacing={10} templateColumns='repeat(auto-fill, minmax(400px, 2fr))'> */}
                  {generatedImages.map((images, index) => (
                      <Card key={index} height="100%" width="90%">
                      <Center>
                        <CardHeader>
                          <Heading size='lg'>{images.title}</Heading>
                        </CardHeader>
                      </Center>
                      <Center>
                        <CardBody mt={-5}>
                          <Center>
                            <Image
                              src={images.image}
                              borderRadius='lg'
                            />
                          </Center>
                        </CardBody>
                      </Center>
                      <Box position='relative' padding='6'>
                        <Divider />
                        <AbsoluteCenter bg='white' px='2'>
                          Classifications
                        </AbsoluteCenter>
                      </Box>
                      <Center>
                        <CardFooter>
                          <VStack spacing="10px">
                            <Center>
                              <SimpleGrid columns={4} spacing={6}>
                                <Button
                                  onClick={() => setSelectedLabel('angry')}
                                  colorScheme={selectedLabel === 'angry' ? 'blue' : 'gray'}
                                  flex="1"
                                >
                                  Angry
                                </Button>
                                <Button
                                  onClick={() => setSelectedLabel('disgust')}
                                  colorScheme={selectedLabel === 'disgust' ? 'blue' : 'gray'}
                                  flex="1"
                                >
                                  Disgust
                                </Button>
                                <Button
                                  onClick={() => setSelectedLabel('fear')}
                                  colorScheme={selectedLabel === 'fear' ? 'blue' : 'gray'}
                                  flex="1"
                                >
                                  Fear
                                </Button>
                                <Button
                                  onClick={() => setSelectedLabel('happy')}
                                  colorScheme={selectedLabel === 'happy' ? 'blue' : 'gray'}
                                  flex="1"
                                >
                                  Happy
                                </Button>
                                <Button
                                  onClick={() => setSelectedLabel('natural')}
                                  colorScheme={selectedLabel === 'natural' ? 'blue' : 'gray'}
                                  flex="1"
                                >
                                  Natural
                                </Button>
                                <Button
                                  onClick={() => setSelectedLabel('sad')}
                                  colorScheme={selectedLabel === 'sad' ? 'blue' : 'gray'}
                                  flex="1"
                                >
                                  Sad
                                </Button>
                                <Button
                                  onClick={() => setSelectedLabel('surprise')}
                                  colorScheme={selectedLabel === 'surprise' ? 'blue' : 'gray'}
                                  flex="1"
                                >
                                  Surprise
                                </Button>
                              </SimpleGrid>
                            </Center>
                          </VStack>
                        </CardFooter>
                      </Center>
                
                      </Card>
                    ))}
               {/* </SimpleGrid> */}
                </Flex>
                <Flex width="30%" justifyContent="center" alignItems="center">
                  <Card height="100%" width="90%">
                    <Center>
                        <CardHeader>
                          <Heading size='lg'>Accuracy</Heading>
                        </CardHeader>
                      </Center>
                    <AbsoluteCenter>
                      <AccuracyChart data={data} />
                    </AbsoluteCenter>
                  </Card>
                </Flex>
              </Flex>
              <Center pt={10} pb={6}>
                <Button onClick={handleClick} colorScheme='blue'> Confirm and Generate New Image </Button>
                <AlertDialog isOpen={isOpen} leastDestructiveRef={cancelRef} onClose={onClose}>
                  <AlertDialogOverlay>
                    <AlertDialogContent>
                      <AlertDialogHeader fontSize="lg" fontWeight="bold">
                        Oops!
                      </AlertDialogHeader>
                      <AlertDialogCloseButton />
                      <AlertDialogBody>
                        {alertMessage}
                      </AlertDialogBody>
                      <AlertDialogFooter>
                        <Button ref={cancelRef} onClick={onClose}>
                          Close
                        </Button>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialogOverlay>
                </AlertDialog>
              </Center>
              <Center>
                <Text fontSize='xs'>
                  Created by M. Barron and A. Shrey
                </Text>
              </Center>
            </TabPanel>

            <TabPanel>
              <Text>
                This model...
              </Text>
              <Center>
                <Text fontSize='xs'>
                  Created by M. Barron and A. Shrey
                </Text>
              </Center>
            </TabPanel>
          </TabPanels>
        </Tabs>

      </ChakraProvider>
  );
}

export default App;



