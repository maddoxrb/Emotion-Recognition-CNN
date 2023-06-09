import {
  React,
  useState,
} from 'react'

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
  Img,
  Card,
  CardBody,
  CardHeader,
  SimpleGrid,
  Divider,
  CardFooter,
  Text,
  AbsoluteCenter,
  Button,
} from '@chakra-ui/react'

import {Helmet} from "react-helmet";

const generatedImages = [
  { title: 'Image 1', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'model'},
  { title: 'Image 2', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'actual'},
  { title: 'Image 3', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'actual'},
  { title: 'Image 4', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'actual'},
  { title: 'Image 5', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'actual'},
  { title: 'Image 6', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'actual'},
  { title: 'Image 7', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'actual'},
  { title: 'Image 8', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'actual'},
  { title: 'Image 9', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'actual'},
  { title: 'Image 10', image: 'https://bit.ly/dan-abramov', model: 'model', actual: 'actual'},
];

function App() {
  const [isButtonClicked, setIsButtonClicked] = useState(false);

  const handleClick = () => {
    setIsButtonClicked(true);
    // Insert python information scrammbler here
    setIsButtonClicked(false);
  };

  return (
    <ChakraProvider>
      <Helmet>
          <title> Face Emotion Detector </title>
          <meta name="Face Emotion Detector" content="A CNN model that determines the emotion of faces." />
      </Helmet>
        <Tabs isFitted variant='line' size='sm'>
          <TabList >
            <Tab>Face Emotion Detector</Tab>
            <Tab >Info</Tab>
          </TabList>

          <TabPanels>
            <TabPanel>
            <VStack spacing='5px'>
                  <Center>
                    <Heading as='h2' size='2xl'>
                      Face Emotion Detector
                    </Heading>
                  </Center>
              </VStack>
              <SimpleGrid m={10} spacing={10} templateColumns='repeat(auto-fill, minmax(400px, 2fr))'>
                    {generatedImages.map((images, index) => (
                      <Card key={index} height="100%" >
                      <Center>
                        <CardHeader>
                          <Heading size='md'>{images.title}</Heading>
                        </CardHeader>
                      </Center>
                      <Center>
                        <CardBody mt={-5}>
                          <Center>
                            <Img
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
                              <VStack>
                                <Heading as='h1' size='md'> Model</Heading>
                                <Text> {images.model} </Text>
                              </VStack>
                            </Center>
                            <Center>
                              <VStack>
                                <Heading as='h1' size='md'> Actual</Heading>
                                <Text> {images.actual} </Text>
                              </VStack>
                            </Center>
                          </VStack>
                        </CardFooter>
                      </Center>
                
                      </Card>
                    ))}
               </SimpleGrid>
               <Center>
                <Button onClick={handleClick} colorScheme='blue'> Scramble </Button>
               </Center>
            </TabPanel>

            <TabPanel>
              <Text>
                This model...
              </Text>
            </TabPanel>
          </TabPanels>
        </Tabs>

      </ChakraProvider>
  );
}

export default App;



