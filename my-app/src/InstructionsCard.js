import { Box, Card, CardBody, ChakraProvider, Text, Flex} from '@chakra-ui/react';
import { useEffect, useRef } from 'react';

const InstructionsCard = () => {
  const boxRef = useRef(null);

  useEffect(() => {
    const updateFlexHeight = () => {
      const boxHeight = boxRef.current.offsetHeight;
      const flexHeight = `${boxHeight + (boxHeight * 0.1)}px`;

      boxRef.current.parentNode.style.height = flexHeight;
    };

    window.addEventListener('resize', updateFlexHeight);
    updateFlexHeight();

    return () => {
      window.removeEventListener('resize', updateFlexHeight);
    };
  }, []);

  return (
    <ChakraProvider>
      <Flex align="center" justify="center" height="100vh" p={10} pt={3}>
        <Box 
          ref={boxRef}
          display="flex" 
          alignItems="center" 
          justifyContent="center"
          width={['80%', '80%', '800px']} // Set the width to 100% for smaller screens, otherwise 400px
          >
          <Card maxW={1200}>
            <CardBody>
              <Text fontSize="lg" mb={4}>
                Welcome to the Flower Classifier! This application uses different CNN models to determine the type of flowers. Follow these instructions to use the application:
                <br /><br />
                Look at the generated images on the right side of the screen. Each image represents a flower. Study the flower and try to determine the corresponding type.
                <br /><br />
                Click on the type buttons below each image to select the type you think best represents the flower. After selecting a type, click the "Generate New Image" button to generate a new flower image.
                <br /><br />
                If you didn't select a type before generating a new image, an alert will remind you to choose one. If you selected the correct type for an image (e.g., "Rose"), the accuracy chart will reflect that.
                <br /><br />
                Use the accuracy chart to track your performance and improve your flower classification skills. You can also see the accuracy of the CNN models along with you.
              </Text>
            </CardBody>
          </Card>
        </Box>
      </Flex>
    </ChakraProvider>
  );
};

export default InstructionsCard;