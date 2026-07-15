using CyclonApp.Repositories.Contracts;
using CyclonApp.Repositories.Repositories;
using Microsoft.AspNetCore.Mvc;

namespace Web.Controllers
{
    public class EngineerController : Controller
    {
        public readonly IUnitOfWork _uow;

        public EngineerController(IUnitOfWork uow)
        {
            _uow = uow;
        }

        public IActionResult Dashboard()
        {
            try
            {
                return View();
            }
            catch (Exception ex)
            {
                _uow.exceptionHandlerRepository.SaveException(
                    "EngineerController",
                    "Dashboard",
                    ex.ToString());

                return RedirectToAction("Index", "Dashboard");
            }
        }

        public IActionResult Projects()
        {
            try
            {
                return View();
            }
            catch (Exception ex)
            {
                _uow.exceptionHandlerRepository.SaveException(
                    "EngineerController",
                    "Projects",
                    ex.ToString());

                return RedirectToAction("Index", "Dashboard");
            }
        }

        public IActionResult Tasks()
        {
            try
            {
                return View();
            }
            catch (Exception ex)
            {
                _uow.exceptionHandlerRepository.SaveException(
                    "EngineerController",
                    "Tasks",
                    ex.ToString());

                return RedirectToAction("Index", "Dashboard");
            }
        }

        public IActionResult Profile()
        {
            try
            {
                return View();
            }
            catch (Exception ex)
            {
                _uow.exceptionHandlerRepository.SaveException(
                    "EngineerController",
                    "Profile",
                    ex.ToString());

                return RedirectToAction("Index", "Dashboard");
            }
        }
    }
}
