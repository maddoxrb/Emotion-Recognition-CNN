import { ChakraProvider, Box, Center, Text, useColorMode } from "@chakra-ui/react";

function Info() {
  const { colorMode } = useColorMode();

  return (
    <ChakraProvider>
      <Box>
        <Center>
            <Text pt={5}> 
                This is a model...
                You can contact us at...
            </Text>
        </Center>
      </Box>
    </ChakraProvider>
  );
}

export default Info;
