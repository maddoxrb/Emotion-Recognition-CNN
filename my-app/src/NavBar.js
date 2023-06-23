import { ChakraProvider, Box, Flex, Spacer, Link, Button, Drawer, DrawerBody, DrawerHeader, DrawerOverlay, DrawerContent, Text, DrawerCloseButton, useDisclosure, useColorMode, IconButton } from "@chakra-ui/react";
import { ChevronDownIcon, HamburgerIcon, MoonIcon, SunIcon } from "@chakra-ui/icons";
import { useState } from "react";

function NavBar() {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const { colorMode, toggleColorMode } = useColorMode();

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <ChakraProvider>
      <Box bg={colorMode === "dark" ? "gray.900" : "gray.200"} p={4}>
        <Flex maxW="auto" mx="auto" align="center">
          <Box>
            <Text as='b' fontSize='3xl'>
              Flower Classifier
            </Text>
          </Box>
          <Spacer />
          <Box display={{ base: "none", md: "block" }} mr={-2}>
            <Link href="model" mx={2} color={colorMode === "dark" ? "white" : "gray.500"} _hover={{ fontWeight: "bold" }}>
              Model Accuracy Testing
            </Link>
            <Link href="/info" mx={2} color={colorMode === "dark" ? "white" : "gray.500"} _hover={{ fontWeight: "bold" }}>
              Info
            </Link>
          </Box>
          <Box display={{ base: "block", md: "none" }}>
            <Button onClick={onOpen} size="sm" variant="ghost" color={colorMode === "dark" ? "white" : "gray.500"}>
              <HamburgerIcon />
            </Button>
          </Box>
          <IconButton
            aria-label="Toggle dark mode"
            icon={colorMode === "dark" ? <SunIcon /> : <MoonIcon />}
            onClick={toggleColorMode}
            ml={2}
            color={colorMode === "dark" ? "white" : "gray.500"}
          />
        </Flex>
      </Box>
      <Box bg="blue.200" height="10px" />
      <Drawer placement="left" onClose={onClose} isOpen={isOpen}>
        <DrawerOverlay>
          <DrawerContent>
            <DrawerCloseButton />
            <DrawerHeader>Menu</DrawerHeader>
            <DrawerBody>
              <Link href="model" display="block" my={2} color={colorMode === "dark" ? "white" : "gray.500"} _hover={{ fontWeight: "bold" }}>
                Model Accuracy Testing
              </Link>
              <Link href="info" display="block" my={2} color={colorMode === "dark" ? "white" : "gray.500"} _hover={{ fontWeight: "bold" }}>
                Info
              </Link>
            </DrawerBody>
          </DrawerContent>
        </DrawerOverlay>
      </Drawer>
    </ChakraProvider>
  );
}

export default NavBar;
